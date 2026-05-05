"""Train the compact residual policy/value baseline."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataset import git_sha, read_jsonl, write_report
from .model import AlphaZeroNet
from .training_data import PositionRecordDataset


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _split_records(records: list, split: str) -> list:
    selected = [r for r in records if r.split == split]
    return selected if selected else records


def train(args: argparse.Namespace, progress_cb=None) -> dict:
    _set_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    records = list(read_jsonl(args.dataset))
    train_records = _split_records(records, "train")
    val_records = [r for r in records if r.split == "validation"]

    train_ds = PositionRecordDataset(train_records)
    val_ds = PositionRecordDataset(val_records)
    if len(train_ds) == 0:
        raise ValueError("no trainable records found; records need move_uci and result")

    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0) if len(val_ds) else None

    net = AlphaZeroNet(args.num_blocks, args.channels, args.dropout_p).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_epoch = 0
    history: list[dict] = []

    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if args.resume and checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        net.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0)
        history = list(ckpt.get("history", []))

    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    for epoch in range(start_epoch + 1, start_epoch + args.epochs + 1):
        net.train()
        sums = {"value_loss": 0.0, "policy_loss": 0.0, "total_loss": 0.0}
        batches = 0
        for xb, policy_idx, legal_mask, value in loader:
            xb = xb.to(device)
            policy_idx = policy_idx.to(device)
            legal_mask = legal_mask.to(device)
            value = value.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                pred_value, logits = net(xb)
                masked_logits = logits.masked_fill(~legal_mask, float("-inf"))
                value_loss = F.mse_loss(pred_value, value)
                policy_loss = F.cross_entropy(masked_logits, policy_idx)
                loss = value_loss + policy_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            sums["value_loss"] += float(value_loss.detach().cpu())
            sums["policy_loss"] += float(policy_loss.detach().cpu())
            sums["total_loss"] += float(loss.detach().cpu())
            batches += 1

        metrics = {k: v / max(batches, 1) for k, v in sums.items()}
        metrics["epoch"] = epoch
        if val_loader is not None:
            metrics.update({f"val_{k}": v for k, v in _evaluate_losses(net, val_loader, device).items()})
        history.append(metrics)
        if progress_cb is not None:
            progress_cb(epoch - start_epoch, args.epochs, metrics)

        torch.save(
            {
                "epoch": epoch,
                "model": net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "history": history,
                "config": vars(args),
            },
            checkpoint_path,
        )

    report = {
        "dataset": str(args.dataset),
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "train_records": len(train_ds),
        "validation_records": len(val_ds),
        "history": history,
        "git_sha": git_sha(Path(__file__).resolve().parents[2]),
    }
    write_report(args.metrics, report)
    write_report(args.config_out, {"config": vars(args), "git_sha": report["git_sha"]})
    return report


def _evaluate_losses(net: AlphaZeroNet, loader: DataLoader, device: torch.device) -> dict:
    net.eval()
    sums = {"value_loss": 0.0, "policy_loss": 0.0, "total_loss": 0.0}
    batches = 0
    with torch.no_grad():
        for xb, policy_idx, legal_mask, value in loader:
            xb = xb.to(device)
            policy_idx = policy_idx.to(device)
            legal_mask = legal_mask.to(device)
            value = value.to(device)
            pred_value, logits = net(xb)
            masked_logits = logits.masked_fill(~legal_mask, float("-inf"))
            value_loss = F.mse_loss(pred_value, value)
            policy_loss = F.cross_entropy(masked_logits, policy_idx)
            loss = value_loss + policy_loss
            sums["value_loss"] += float(value_loss.cpu())
            sums["policy_loss"] += float(policy_loss.cpu())
            sums["total_loss"] += float(loss.cpu())
            batches += 1
    return {k: v / max(batches, 1) for k, v in sums.items()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--config-out", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--dropout-p", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--amp", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    train(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
