import argparse

import chess

from chess_resurected.dataset import stable_position_id, write_jsonl
from chess_resurected.records import PositionRecord
from chess_resurected.train_baseline import train


def _record(index: int, fen: str, move: str, result: float) -> PositionRecord:
    return PositionRecord(
        position_id=stable_position_id(f"game-{index}", index, fen, move),
        fen=fen,
        move_uci=move,
        legal_moves_uci=tuple(m.uci() for m in chess.Board(fen).legal_moves),
        result=result,
        source_game_id=f"game-{index}",
        ply=index,
        split="train",
    )


def test_train_baseline_writes_checkpoint_and_metrics(tmp_path):
    dataset = tmp_path / "records.jsonl"
    checkpoint = tmp_path / "model.pt"
    metrics = tmp_path / "metrics.json"
    config = tmp_path / "config.json"
    rows = [
        _record(0, chess.STARTING_FEN, "e2e4", 1.0),
        _record(1, chess.STARTING_FEN, "d2d4", 1.0),
        _record(2, chess.STARTING_FEN, "g1f3", 0.0),
        _record(3, chess.STARTING_FEN, "c2c4", -1.0),
    ]
    write_jsonl(dataset, rows)

    report = train(
        argparse.Namespace(
            dataset=str(dataset),
            checkpoint=str(checkpoint),
            metrics=str(metrics),
            config_out=str(config),
            epochs=1,
            batch_size=2,
            lr=1e-3,
            weight_decay=1e-4,
            num_blocks=1,
            channels=8,
            dropout_p=0.0,
            seed=1,
            device="cpu",
            resume=False,
            amp=False,
        )
    )

    assert checkpoint.exists()
    assert metrics.exists()
    assert config.exists()
    assert report["train_records"] == 4
    assert report["history"][0]["total_loss"] > 0
