"""JSONL dataset helpers and lifecycle reports."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator

from .records import DatasetSplit, PositionRecord


def stable_position_id(source_game_id: str, ply: int, fen: str, move_uci: str | None) -> str:
    """Return a deterministic ID for a source-game ply."""
    h = hashlib.sha256()
    h.update(source_game_id.encode("utf-8"))
    h.update(b"\0")
    h.update(str(ply).encode("ascii"))
    h.update(b"\0")
    h.update(fen.encode("utf-8"))
    h.update(b"\0")
    h.update((move_uci or "").encode("ascii"))
    return h.hexdigest()[:32]


def assign_split(source_game_id: str, train_pct: int = 80, validation_pct: int = 10) -> DatasetSplit:
    """Deterministically assign a game to train/validation/test."""
    bucket = int(hashlib.sha256(source_game_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < train_pct:
        return "train"
    if bucket < train_pct + validation_pct:
        return "validation"
    return "test"


def read_jsonl(path: str | Path) -> Iterator[PositionRecord]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield PositionRecord.from_json_dict(json.loads(line))
            except Exception as exc:
                raise ValueError(f"{path}:{line_no}: invalid PositionRecord JSON: {exc}") from exc


def write_jsonl(path: str | Path, records: Iterable[PositionRecord]) -> int:
    count = 0
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record.to_json_dict(), sort_keys=True, separators=(",", ":")))
            f.write("\n")
            count += 1
    return count


def git_sha(repo_root: str | Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            text=True,
            capture_output=True,
        )
    except Exception:
        return None
    return result.stdout.strip()


def dataset_report(records: list[PositionRecord], *, input_count: int | None = None) -> dict:
    count = len(records)
    split_counts = Counter(r.split for r in records)
    result_counts = Counter("none" if r.result is None else str(r.result) for r in records)
    plies = [r.ply for r in records if r.ply is not None]
    duplicate_keys = count - len({(r.fen, r.move_uci) for r in records})
    illegal = sum(
        1
        for r in records
        if r.move_uci is not None and r.legal_moves_uci and r.move_uci not in r.legal_moves_uci
    )

    return {
        "input_records": input_count if input_count is not None else count,
        "output_records": count,
        "removed_records": max((input_count if input_count is not None else count) - count, 0),
        "dedup_removed": duplicate_keys,
        "illegal_records": illegal,
        "illegal_rate": illegal / count if count else 0.0,
        "split_counts": dict(sorted(split_counts.items())),
        "result_counts": dict(sorted(result_counts.items())),
        "avg_ply": sum(plies) / len(plies) if plies else 0.0,
    }


def write_report(path: str | Path, report: dict) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
