import argparse

import chess

from chess_resurected.dataset import assign_split, read_jsonl, stable_position_id, write_jsonl
from chess_resurected.filter_dataset import filter_records
from chess_resurected.records import PositionRecord
from chess_resurected.training_data import PositionRecordDataset
from chess_resurected.validation import validate_record


START = chess.STARTING_FEN


class FakeOracle:
    def legal_moves(self, _fen: str):
        return ("e2e4", "g1f3")

    def is_legal(self, _fen: str, move_uci: str) -> bool:
        return move_uci in self.legal_moves(_fen)


def record(move: str = "e2e4") -> PositionRecord:
    return PositionRecord(
        position_id=stable_position_id("game", 0, START, move),
        fen=START,
        move_uci=move,
        legal_moves_uci=("e2e4", "g1f3"),
        result=1.0,
        source_game_id="game",
        ply=0,
        split="train",
    )


def test_jsonl_roundtrip_and_split_are_deterministic(tmp_path):
    path = tmp_path / "records.jsonl"
    rows = [record()]

    assert write_jsonl(path, rows) == 1
    assert list(read_jsonl(path)) == rows
    assert assign_split("stable-game") == assign_split("stable-game")


def test_validation_catches_bad_uci_and_illegal_move():
    bad = record("e2e9")
    illegal = record("a1a2")

    assert any(issue.field == "move_uci" for issue in validate_record(bad))
    assert any(
        issue.field == "move_uci"
        for issue in validate_record(illegal, oracle=FakeOracle(), require_legal=True)
    )


def test_filter_records_dedups_and_removes_illegal():
    rows = [record("e2e4"), record("e2e4"), record("a1a2")]
    kept, report = filter_records(rows, oracle=FakeOracle())

    assert [r.move_uci for r in kept] == ["e2e4"]
    assert report["dedup_removed"] == 1
    assert report["illegal_removed"] == 1


def test_training_dataset_shapes():
    ds = PositionRecordDataset([record("e2e4")])
    tensor, policy_idx, legal_mask, value = ds[0]

    assert tensor.shape == (18, 8, 8)
    assert policy_idx.item() >= 0
    assert legal_mask.shape == (4096,)
    assert value.item() == 1.0
