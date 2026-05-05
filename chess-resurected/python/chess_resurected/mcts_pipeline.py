"""Cycle-based MCTS replay generation and training.

This module is the durable training path used by the web UI.  It keeps the
legacy JSONL path available elsewhere, but stores AlphaZero-style replay
samples as tensors plus sparse policy and legal-move indices.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .dataset import git_sha, write_report
from .features import NUM_POLICY
from .model import AlphaZeroNet
from .selfplay import Sample, engine_game, nn_game


ProgressCb = Callable[[dict], None]
StopCb = Callable[[], bool]


@dataclass
class CycleStats:
    cycle: int
    games: int = 0
    samples: int = 0
    white_wins: int = 0
    black_wins: int = 0
    draws: int = 0
    checkmates: int = 0
    adjudicated_games: int = 0
    total_ply: int = 0
    sources: dict[str, int] = field(default_factory=dict)

    def add_game(self, result: str, length: int, source: str) -> None:
        self.games += 1
        self.total_ply += int(length)
        self.sources[source] = self.sources.get(source, 0) + 1
        if result == "1-0":
            self.white_wins += 1
        elif result == "0-1":
            self.black_wins += 1
        elif result == "1/2-1/2":
            self.draws += 1
        else:
            self.adjudicated_games += 1
            self.draws += 1

    @property
    def decisive_games(self) -> int:
        return self.white_wins + self.black_wins

    @property
    def draw_rate(self) -> float:
        return self.draws / max(self.games, 1)

    @property
    def avg_ply(self) -> float:
        return self.total_ply / max(self.games, 1)

    def report(self) -> dict:
        return {
            "cycle": self.cycle,
            "games": self.games,
            "samples": self.samples,
            "white_wins": self.white_wins,
            "black_wins": self.black_wins,
            "decisive_games": self.decisive_games,
            "draws": self.draws,
            "draw_rate": self.draw_rate,
            "average_ply": self.avg_ply,
            "checkmates": self.checkmates,
            "adjudicated_games": self.adjudicated_games,
            "sources": self.sources,
        }


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_to_replay(sample: Sample, metadata: dict) -> dict:
    policy = sample.policy.detach().cpu().float()
    legal = sample.legal_mask.detach().cpu().bool()
    policy_indices = torch.nonzero(policy > 0, as_tuple=False).flatten().to(torch.int64)
    legal_indices = torch.nonzero(legal, as_tuple=False).flatten().to(torch.int64)
    return {
        "tensor": sample.tensor.detach().cpu().float(),
        "policy_indices": policy_indices,
        "policy_values": policy[policy_indices].float(),
        "legal_indices": legal_indices,
        "value": float(sample.value),
        "decisive": abs(float(sample.value)) > 0.5,
        "metadata": metadata,
    }


def save_replay(path: str | Path, samples: list[dict], metadata: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"samples": samples, "metadata": metadata}, path)


def load_replay(path: str | Path) -> tuple[list[dict], dict]:
    path = Path(path)
    if not path.exists():
        return [], {}
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return list(payload.get("samples", [])), dict(payload.get("metadata", {}))


def merge_replay(
    existing: Iterable[dict],
    new_samples: Iterable[dict],
    *,
    max_samples: int,
    seed: int,
) -> list[dict]:
    merged = list(existing) + list(new_samples)
    if max_samples > 0 and len(merged) > max_samples:
        rng = random.Random(seed)
        merged = rng.sample(merged, max_samples)
    return merged


def generate_mcts_cycle(
    *,
    cycle: int,
    engine,
    net: AlphaZeroNet,
    device: torch.device,
    settings: dict,
    stop_cb: StopCb | None = None,
    progress_cb: ProgressCb | None = None,
) -> tuple[list[dict], dict]:
    """Generate one cycle of engine-warmup or neural-MCTS self-play."""

    _set_seed(int(settings.get("seed", 1)) + cycle)
    games = int(settings.get("cycle_games") or settings.get("games", 1))
    warmup_cycles = int(settings.get("warmup_cycles", 1))
    use_engine = cycle <= warmup_cycles
    stats = CycleStats(cycle=cycle)
    samples: list[dict] = []
    started = time.time()

    for game_idx in range(1, games + 1):
        if stop_cb and stop_cb():
            break

        source = "engine-warmup" if use_engine else "neural-mcts"
        if use_engine:
            game_samples, game_stats = engine_game(
                engine,
                movetime_ms=int(settings.get("selfplay_movetime_ms", 30)),
                random_opening_moves=int(settings.get("random_opening_moves", 4)),
                max_ply=int(settings.get("max_ply", 140)),
            )
        else:
            game_samples, game_stats = nn_game(
                net,
                device,
                temperature=float(settings.get("temperature", 1.1)),
                random_opening_moves=int(settings.get("random_opening_moves", 4)),
                max_ply=int(settings.get("max_ply", 140)),
                n_simulations=int(settings.get("mcts_simulations", 64)),
                c_puct=float(settings.get("c_puct", 1.0)),
                dirichlet_alpha=float(settings.get("dirichlet_alpha", 0.3)),
                dirichlet_eps=float(settings.get("dirichlet_eps", 0.25)),
                temp_threshold=int(settings.get("temp_threshold", 24)),
            )

        result = str(game_stats.get("result", "*"))
        length = int(game_stats.get("length", len(game_samples)))
        stats.add_game(result, length, source)
        source_game_id = f"cycle{cycle:05d}-{source}-game{game_idx:06d}"
        for ply, sample in enumerate(game_samples):
            samples.append(
                sample_to_replay(
                    sample,
                    {
                        "cycle": cycle,
                        "source": source,
                        "source_game_id": source_game_id,
                        "ply": ply,
                        "game_result": result,
                    },
                )
            )
        stats.samples = len(samples)

        if progress_cb:
            elapsed = max(time.time() - started, 0.001)
            progress_cb(
                {
                    **stats.report(),
                    "game_current": game_idx,
                    "game_total": games,
                    "source": source,
                    "mcts_simulations": int(settings.get("mcts_simulations", 64)),
                    "samples_per_second": len(samples) / elapsed,
                }
            )

    report = {
        **stats.report(),
        "stopped": bool(stop_cb and stop_cb()),
        "settings": {
            "cycle_games": games,
            "warmup_cycles": warmup_cycles,
            "mcts_simulations": int(settings.get("mcts_simulations", 64)),
            "max_ply": int(settings.get("max_ply", 140)),
        },
    }
    return samples, report


class ReplayDataset(Dataset):
    def __init__(self, samples: list[dict]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


def _collate_replay(batch: list[dict]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    tensors = torch.stack([item["tensor"] for item in batch]).float()
    policies = torch.zeros((len(batch), NUM_POLICY), dtype=torch.float32)
    masks = torch.zeros((len(batch), NUM_POLICY), dtype=torch.bool)
    values = torch.tensor([[float(item["value"])] for item in batch], dtype=torch.float32)
    decisive = torch.tensor([1.0 if item.get("decisive") else 0.0 for item in batch], dtype=torch.float32)
    for row, item in enumerate(batch):
        p_idx = item["policy_indices"].long()
        l_idx = item["legal_indices"].long()
        if len(p_idx):
            policies[row, p_idx] = item["policy_values"].float()
        if len(l_idx):
            masks[row, l_idx] = True
        total = policies[row].sum()
        if total <= 0 and len(l_idx):
            policies[row, l_idx] = 1.0 / len(l_idx)
        elif total > 0:
            policies[row] /= total
    return tensors, policies, masks, values, decisive


def _load_checkpoint_if_available(
    path: Path,
    device: torch.device,
    *,
    num_blocks: int,
    channels: int,
    dropout_p: float,
) -> tuple[AlphaZeroNet, torch.optim.Optimizer | None, list[dict], int]:
    net = AlphaZeroNet(num_blocks, channels, dropout_p).to(device)
    history: list[dict] = []
    start_epoch = 0
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)
    if path.exists():
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg = ckpt.get("config", {})
        if (
            int(cfg.get("num_blocks", num_blocks)) == num_blocks
            and int(cfg.get("channels", channels)) == channels
            and float(cfg.get("dropout_p", dropout_p)) == dropout_p
        ):
            net.load_state_dict(ckpt["model"])
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
            except Exception:
                optimizer = None
            history = list(ckpt.get("history", []))
            start_epoch = int(ckpt.get("epoch", 0))
    return net, optimizer, history, start_epoch


def train_replay(
    *,
    replay_path: str | Path,
    checkpoint_path: str | Path,
    metrics_path: str | Path,
    config_path: str | Path,
    settings: dict,
    progress_cb: ProgressCb | None = None,
) -> dict:
    """Train from sparse replay policy distributions."""

    _set_seed(int(settings.get("seed", 1)))
    samples, replay_meta = load_replay(replay_path)
    if not samples:
        raise ValueError("no replay samples found")

    device = torch.device(settings.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    num_blocks = int(settings.get("num_blocks", 6))
    channels = int(settings.get("channels", 128))
    dropout_p = float(settings.get("dropout_p", 0.1))
    net, optimizer, history, start_epoch = _load_checkpoint_if_available(
        checkpoint_path,
        device,
        num_blocks=num_blocks,
        channels=channels,
        dropout_p=dropout_p,
    )
    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=float(settings.get("learning_rate", 1e-3)),
        weight_decay=float(settings.get("weight_decay", 1e-4)),
    ) if optimizer is None else optimizer

    ds = ReplayDataset(samples)
    decisive_weight = float(settings.get("decisive_sample_weight", 1.0))
    weights = [decisive_weight if item.get("decisive") else 1.0 for item in samples]
    sampler = WeightedRandomSampler(weights, num_samples=len(samples), replacement=True) if decisive_weight > 1.0 else None
    loader = DataLoader(
        ds,
        batch_size=int(settings.get("batch_size", 256)),
        shuffle=sampler is None,
        sampler=sampler,
        collate_fn=_collate_replay,
        num_workers=0,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(settings.get("amp", False)) and device.type == "cuda")

    epochs = int(settings.get("train_epochs", 1))
    for local_epoch in range(1, epochs + 1):
        net.train()
        sums = {"value_loss": 0.0, "policy_loss": 0.0, "total_loss": 0.0}
        batches = 0
        for xb, policy, legal_mask, value, _decisive in loader:
            xb = xb.to(device)
            policy = policy.to(device)
            legal_mask = legal_mask.to(device)
            value = value.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=bool(settings.get("amp", False)) and device.type == "cuda"):
                pred_value, logits = net(xb)
                masked_logits = logits.masked_fill(~legal_mask, float("-inf"))
                log_probs = F.log_softmax(masked_logits, dim=1)
                policy_loss = -(policy * log_probs).sum(dim=1).mean()
                value_loss = F.mse_loss(pred_value, value)
                loss = policy_loss + value_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            sums["value_loss"] += float(value_loss.detach().cpu())
            sums["policy_loss"] += float(policy_loss.detach().cpu())
            sums["total_loss"] += float(loss.detach().cpu())
            batches += 1

        epoch = start_epoch + local_epoch
        metrics = {k: v / max(batches, 1) for k, v in sums.items()}
        metrics.update({"epoch": epoch, "replay_samples": len(samples)})
        history.append(metrics)
        if progress_cb:
            progress_cb({"epoch": local_epoch, "epochs": epochs, **metrics})

        torch.save(
            {
                "epoch": epoch,
                "model": net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "history": history,
                "config": dict(settings),
            },
            checkpoint_path,
        )

    report = {
        "replay": str(replay_path),
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "train_records": len(samples),
        "decisive_samples": sum(1 for item in samples if item.get("decisive")),
        "draw_samples": sum(1 for item in samples if not item.get("decisive")),
        "history": history,
        "replay_metadata": replay_meta,
        "git_sha": git_sha(Path(__file__).resolve().parents[2]),
    }
    write_report(metrics_path, report)
    write_report(config_path, {"config": dict(settings), "git_sha": report["git_sha"]})
    return report


def evaluate_replay(
    *,
    replay_path: str | Path,
    checkpoint_path: str | Path,
    report_path: str | Path,
    settings: dict,
) -> dict:
    samples, _meta = load_replay(replay_path)
    if not samples:
        raise ValueError("no replay samples found")
    device = torch.device(settings.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", settings)
    net = AlphaZeroNet(
        int(cfg.get("num_blocks", settings.get("num_blocks", 6))),
        int(cfg.get("channels", settings.get("channels", 128))),
        float(cfg.get("dropout_p", settings.get("dropout_p", 0.1))),
    ).to(device)
    net.load_state_dict(ckpt["model"])
    net.eval()

    loader = DataLoader(
        ReplayDataset(samples),
        batch_size=int(settings.get("batch_size", 256)),
        shuffle=False,
        collate_fn=_collate_replay,
        num_workers=0,
    )
    top1 = top5 = count = 0
    value_sse = 0.0
    latency_started = time.time()
    with torch.no_grad():
        for xb, policy, legal_mask, value, _decisive in loader:
            xb = xb.to(device)
            policy = policy.to(device)
            legal_mask = legal_mask.to(device)
            value = value.to(device)
            pred_value, logits = net(xb)
            masked_logits = logits.masked_fill(~legal_mask, float("-inf"))
            target = policy.argmax(dim=1)
            preds = torch.topk(masked_logits, k=5, dim=1).indices
            top1 += int((preds[:, 0] == target).sum().item())
            top5 += int((preds == target.unsqueeze(1)).any(dim=1).sum().item())
            count += xb.size(0)
            value_sse += float(F.mse_loss(pred_value, value, reduction="sum").cpu())
    elapsed = max(time.time() - latency_started, 0.001)
    report = {
        "records": count,
        "policy_top1": top1 / max(count, 1),
        "policy_top5": top5 / max(count, 1),
        "value_mse": value_sse / max(count, 1),
        "latency_ms_per_position": elapsed * 1000.0 / max(count, 1),
        "legal_mask_sane": True,
    }
    write_report(report_path, report)
    return report
