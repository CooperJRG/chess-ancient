"""Validation helpers for position records."""

from __future__ import annotations

import re
from dataclasses import dataclass

import chess

from .engine_bridge import LegalMoveOracle
from .records import PositionRecord

UCI_RE = re.compile(r"^[a-h][1-8][a-h][1-8][qrbn]?$")


@dataclass(frozen=True)
class ValidationIssue:
    field: str
    message: str


def validate_fen(fen: str) -> list[ValidationIssue]:
    try:
        chess.Board(fen)
    except ValueError as exc:
        return [ValidationIssue("fen", str(exc))]
    return []


def validate_uci(move_uci: str | None) -> list[ValidationIssue]:
    if move_uci is None:
        return []
    if not UCI_RE.match(move_uci):
        return [ValidationIssue("move_uci", f"invalid UCI move: {move_uci}")]
    return []


def validate_record(
    record: PositionRecord,
    *,
    oracle: LegalMoveOracle | None = None,
    require_legal: bool = False,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    issues.extend(validate_fen(record.fen))
    issues.extend(validate_uci(record.move_uci))

    if record.split not in {"train", "validation", "test", "unsplit"}:
        issues.append(ValidationIssue("split", f"invalid split: {record.split}"))

    if record.result is not None and record.result not in {-1.0, 0.0, 1.0}:
        issues.append(ValidationIssue("result", "result must be -1.0, 0.0, 1.0, or None"))

    if record.move_uci and record.legal_moves_uci and record.move_uci not in record.legal_moves_uci:
        issues.append(ValidationIssue("move_uci", "move is not present in legal_moves_uci"))

    if require_legal and oracle is not None and record.move_uci:
        try:
            if not oracle.is_legal(record.fen, record.move_uci):
                issues.append(ValidationIssue("move_uci", "move is illegal according to Rust engine"))
        except Exception as exc:
            issues.append(ValidationIssue("engine", str(exc)))

    return issues


def assert_valid_record(record: PositionRecord, **kwargs: object) -> None:
    issues = validate_record(record, **kwargs)
    if issues:
        details = "; ".join(f"{i.field}: {i.message}" for i in issues)
        raise ValueError(details)
