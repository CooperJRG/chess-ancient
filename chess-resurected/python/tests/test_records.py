from chess_resurected import PositionRecord


def test_position_record_round_trips_through_json_dict() -> None:
    record = PositionRecord(
        position_id="startpos",
        fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        move_uci="e2e4",
        legal_moves_uci=("e2e4", "d2d4"),
        result=1.0,
        source_game_id="demo",
        ply=0,
        split="train",
        metadata={"source": "unit-test"},
    )

    encoded = record.to_json_dict()
    decoded = PositionRecord.from_json_dict(encoded)

    assert decoded == record
    assert encoded["legal_moves_uci"] == ["e2e4", "d2d4"]
