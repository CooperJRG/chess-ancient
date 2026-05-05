"""Generate self-play PositionRecord JSONL datasets."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import chess

from .dataset import assign_split, dataset_report, git_sha, stable_position_id, write_jsonl, write_report
from .engine_bridge import LegalMoveOracle, UciEngine
from .records import PositionRecord


def _result_for_color(result: str, color: chess.Color) -> float:
    if result == "1-0":
        return 1.0 if color == chess.WHITE else -1.0
    if result == "0-1":
        return 1.0 if color == chess.BLACK else -1.0
    return 0.0


def _choose_move(
    board: chess.Board,
    rng: random.Random,
    *,
    mode: str,
    engine: UciEngine | None,
    movetime_ms: int,
    nodes: int | None,
) -> chess.Move | None:
    if mode == "engine":
        if engine is None:
            raise ValueError("engine mode requires an engine")
        uci = engine.bestmove(board.fen(), movetime_ms=movetime_ms, nodes=nodes)
        if uci is None:
            return None
        move = chess.Move.from_uci(uci)
        return move if move in board.legal_moves else None
    legal = list(board.legal_moves)
    return rng.choice(legal) if legal else None


def generate_selfplay_records(
    *,
    games: int,
    max_ply: int,
    seed: int,
    oracle: LegalMoveOracle,
    split: str | None = None,
    mode: str = "random",
    engine_path: str | None = None,
    movetime_ms: int = 50,
    nodes: int | None = None,
) -> list[PositionRecord]:
    rng = random.Random(seed)
    records: list[PositionRecord] = []
    uci_engine = UciEngine(engine_path or oracle.engine_path) if mode == "engine" else None

    try:
        for game_idx in range(games):
            board = chess.Board()
            game_id = f"selfplay-{mode}-seed{seed}-game{game_idx:06d}"
            pending: list[tuple[str, str, tuple[str, ...], chess.Color, int]] = []

            for ply in range(max_ply):
                if board.is_game_over(claim_draw=True):
                    break
                fen = board.fen()
                legal_moves = oracle.legal_moves(fen)
                if not legal_moves:
                    break
                move = _choose_move(
                    board,
                    rng,
                    mode=mode,
                    engine=uci_engine,
                    movetime_ms=movetime_ms,
                    nodes=nodes,
                )
                if move is None:
                    break
                move_uci = move.uci()
                pending.append((fen, move_uci, legal_moves, board.turn, ply))
                board.push(move)

            result = board.result(claim_draw=True)
            assigned_split = split or assign_split(game_id)
            for fen, move_uci, legal_moves, turn, ply in pending:
                records.append(
                    PositionRecord(
                        position_id=stable_position_id(game_id, ply, fen, move_uci),
                        fen=fen,
                        move_uci=move_uci,
                        legal_moves_uci=legal_moves,
                        result=_result_for_color(result, turn),
                        source_game_id=game_id,
                        ply=ply,
                        split=assigned_split,  # type: ignore[arg-type]
                        metadata={"generator": "selfplay", "mode": mode, "game_result": result},
                    )
                )
    finally:
        if uci_engine is not None:
            uci_engine.close()

    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="Output PositionRecord JSONL")
    parser.add_argument("--report", help="Optional report JSON path")
    parser.add_argument("--games", type=int, default=2)
    parser.add_argument("--max-ply", type=int, default=120)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--engine", help="Path to chess-resurected engine binary")
    parser.add_argument("--split", choices=["train", "validation", "test", "unsplit"])
    parser.add_argument("--mode", choices=["random", "engine"], default="random")
    parser.add_argument("--movetime-ms", type=int, default=50)
    parser.add_argument("--nodes", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    oracle = LegalMoveOracle(args.engine)
    records = generate_selfplay_records(
        games=args.games,
        max_ply=args.max_ply,
        seed=args.seed,
        oracle=oracle,
        split=args.split,
        mode=args.mode,
        engine_path=args.engine,
        movetime_ms=args.movetime_ms,
        nodes=args.nodes,
    )
    count = write_jsonl(args.output, records)
    if args.report:
        report = dataset_report(records, input_count=count)
        report.update(
            {
                "output_path": str(args.output),
                "games": args.games,
                "max_ply": args.max_ply,
                "seed": args.seed,
                "mode": args.mode,
                "git_sha": git_sha(Path(__file__).resolve().parents[2]),
            }
        )
        write_report(args.report, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
