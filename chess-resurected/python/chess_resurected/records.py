"""Stable position-record schema for ingestion and training pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

DatasetSplit = Literal["train", "validation", "test", "unsplit"]


@dataclass(frozen=True, slots=True)
class PositionRecord:
    """A normalized, JSON-serializable chess position record.

    The schema mirrors `docs/resurrection-plan.md` at the repository root and
    `chess-resurected/docs/architecture.md` in the migrated workspace. PGN
    ingestion, generated self-play, and future engine analysis should emit this
    shape before training code consumes positions.
    """

    position_id: str
    fen: str
    move_uci: str | None = None
    legal_moves_uci: tuple[str, ...] = ()
    result: float | None = None
    score_cp: int | None = None
    mate_in: int | None = None
    source_game_id: str | None = None
    ply: int | None = None
    split: DatasetSplit = "unsplit"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary with stable field names."""
        return {
            "position_id": self.position_id,
            "fen": self.fen,
            "move_uci": self.move_uci,
            "legal_moves_uci": list(self.legal_moves_uci),
            "result": self.result,
            "score_cp": self.score_cp,
            "mate_in": self.mate_in,
            "source_game_id": self.source_game_id,
            "ply": self.ply,
            "split": self.split,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_json_dict(cls, value: dict[str, Any]) -> "PositionRecord":
        """Create a record from a JSON dictionary."""
        return cls(
            position_id=value["position_id"],
            fen=value["fen"],
            move_uci=value.get("move_uci"),
            legal_moves_uci=tuple(value.get("legal_moves_uci", ())),
            result=value.get("result"),
            score_cp=value.get("score_cp"),
            mate_in=value.get("mate_in"),
            source_game_id=value.get("source_game_id"),
            ply=value.get("ply"),
            split=value.get("split", "unsplit"),
            metadata=dict(value.get("metadata", {})),
        )
