"""Evaluate a baseline checkpoint on PositionRecord datasets."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataset import git_sha, read_jsonl, write_report
from .model import AlphaZeroNet
from .training_data import PositionRecordDataset


TACTICAL_SMOKE_FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
    "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
    "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2",
]


def _checkpoint_config(ckpt: dict) -> dict:
    config = ckpt.get("config", {})
    return {
        "num_blocks": int(config.get("num_blocks", 4)),
        "channels": int(config.get("channels", 64)),
        "dropout_p": float(config.get("dropout_p", 0.1)),
    }


def evaluate(args: argparse.Namespace) -> dict:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    net = AlphaZeroNet(**_checkpoint_config(ckpt)).to(device)
    net.load_state_dict(ckpt["model"])
    net.eval()

    records = list(read_jsonl(args.dataset))
    selected = [r for r in records if r.split == args.split] if args.split else records
    ds = PositionRecordDataset(selected)
    if len(ds) == 0:
        raise ValueError("no evaluable records found")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    total = 0
    top1 = 0
    topk = 0
    illegal_argmax = 0
    value_se = 0.0
    policy_loss_sum = 0.0
    start = time.perf_counter()
    with torch.no_grad():
        for xb, policy_idx, legal_mask, value in loader:
            xb = xb.to(device)
            policy_idx = policy_idx.to(device)
            legal_mask = legal_mask.to(device)
            value = value.to(device)
            pred_value, logits = net(xb)
            masked_logits = logits.masked_fill(~legal_mask, float("-inf"))
            pred = masked_logits.argmax(dim=1)
            k = min(args.top_k, masked_logits.shape[1])
            top = masked_logits.topk(k, dim=1).indices
            top1 += int((pred == policy_idx).sum().cpu())
            topk += int((top == policy_idx.unsqueeze(1)).any(dim=1).sum().cpu())
            illegal_argmax += int((~legal_mask.gather(1, pred.unsqueeze(1))).sum().cpu())
            value_se += float(F.mse_loss(pred_value, value, reduction="sum").cpu())
            policy_loss_sum += float(F.cross_entropy(masked_logits, policy_idx, reduction="sum").cpu())
            total += xb.shape[0]
    elapsed = time.perf_counter() - start

    report = {
        "dataset": str(args.dataset),
        "checkpoint": str(args.checkpoint),
        "split": args.split or "all",
        "records": total,
        "policy_top1": top1 / total,
        f"policy_top{args.top_k}": topk / total,
        "value_mse": value_se / total,
        "policy_loss": policy_loss_sum / total,
        "illegal_argmax": illegal_argmax,
        "positions_per_second": total / elapsed if elapsed > 0 else 0.0,
        "latency_ms_per_position": (elapsed / total) * 1000 if total else 0.0,
        "tactical_smoke_fens": TACTICAL_SMOKE_FENS,
        "device": str(device),
        "git_sha": git_sha(Path(__file__).resolve().parents[2]),
    }
    write_report(args.report, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--split", choices=["train", "validation", "test"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", choices=["cpu", "cuda"])
    return parser


def main(argv: list[str] | None = None) -> int:
    evaluate(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
