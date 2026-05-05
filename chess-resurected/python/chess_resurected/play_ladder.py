"""Play a trained model against a configurable external UCI engine."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import chess
import chess.pgn
import torch

from .dataset import git_sha, write_report
from .engine_bridge import EngineBridgeError, UciEngine
from .features import board_to_tensor, move_to_index
from .model import AlphaZeroNet

OPENINGS = [
    [],
    ["e2e4", "e7e5", "g1f3", "b8c6"],
    ["d2d4", "d7d5", "c2c4"],
    ["c2c4", "e7e5", "g1c3"],
    ["g1f3", "d7d5", "g2g3"],
]


def _checkpoint_config(ckpt: dict) -> dict:
    config = ckpt.get("config", {})
    return {
        "num_blocks": int(config.get("num_blocks", 4)),
        "channels": int(config.get("channels", 64)),
        "dropout_p": float(config.get("dropout_p", 0.1)),
    }


def _load_model(path: str, device: torch.device) -> AlphaZeroNet:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    net = AlphaZeroNet(**_checkpoint_config(ckpt)).to(device)
    net.load_state_dict(ckpt["model"])
    net.eval()
    return net


def _model_move(net: AlphaZeroNet, board: chess.Board, device: torch.device) -> chess.Move | None:
    legal = list(board.legal_moves)
    if not legal:
        return None
    with torch.no_grad():
        _value, logits = net(board_to_tensor(board).unsqueeze(0).to(device))
    logits = logits.squeeze(0)
    return max(legal, key=lambda move: float(logits[move_to_index(move, board)]))


def _apply_opening(board: chess.Board, opening: list[str]) -> None:
    for uci in opening:
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            return
        board.push(move)


def _elo_interval(score: float, games: int, opponent_elo: float) -> tuple[float, float, float]:
    eps = 1e-6
    score = min(max(score, eps), 1.0 - eps)
    se = math.sqrt(score * (1.0 - score) / max(games, 1))
    lower_score = min(max(score - 1.96 * se, eps), 1.0 - eps)
    upper_score = min(max(score + 1.96 * se, eps), 1.0 - eps)

    def convert(s: float) -> float:
        return opponent_elo + 400.0 * math.log10(s / (1.0 - s))

    return convert(score), convert(lower_score), convert(upper_score)


def play_ladder(args: argparse.Namespace, progress_cb=None) -> dict:
    opponent_path = args.opponent_engine or os.environ.get("STOCKFISH_PATH")
    if not opponent_path:
        raise EngineBridgeError(
            "No external opponent configured. Pass --opponent-engine or set STOCKFISH_PATH."
        )

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    net = _load_model(args.checkpoint, device)
    wins = draws = losses = 0
    pgn_games: list[chess.pgn.Game] = []

    with UciEngine(opponent_path) as opponent:
        for game_idx in range(args.games):
            board = chess.Board()
            opening = OPENINGS[game_idx % len(OPENINGS)]
            _apply_opening(board, opening)
            model_white = game_idx % 2 == 0
            game = chess.pgn.Game()
            game.headers["Event"] = "Chess Resurected ladder"
            game.headers["White"] = "Chess Resurected model" if model_white else "External UCI"
            game.headers["Black"] = "External UCI" if model_white else "Chess Resurected model"
            game.headers["Round"] = str(game_idx + 1)
            node = game
            for move in board.move_stack:
                node = node.add_variation(move)

            for _ in range(args.max_ply):
                if board.is_game_over(claim_draw=True):
                    break
                model_turn = (board.turn == chess.WHITE) == model_white
                if model_turn:
                    move = _model_move(net, board, device)
                else:
                    uci = opponent.bestmove(
                        board.fen(),
                        movetime_ms=args.movetime_ms,
                        nodes=args.nodes,
                    )
                    move = chess.Move.from_uci(uci) if uci else None
                    if move not in board.legal_moves:
                        move = None
                if move is None:
                    break
                board.push(move)
                node = node.add_variation(move)

            result = board.result(claim_draw=True)
            game.headers["Result"] = result
            pgn_games.append(game)
            model_win = "1-0" if model_white else "0-1"
            model_loss = "0-1" if model_white else "1-0"
            if result == model_win:
                wins += 1
            elif result == model_loss:
                losses += 1
            else:
                draws += 1
            if progress_cb is not None:
                progress_cb(game_idx + 1, args.games, {"wins": wins, "draws": draws, "losses": losses})

    score = (wins + 0.5 * draws) / max(args.games, 1)
    elo, elo_low, elo_high = _elo_interval(score, args.games, args.opponent_elo)
    accepted = elo_low >= args.fm_target
    report = {
        "checkpoint": str(args.checkpoint),
        "opponent_engine": str(opponent_path),
        "opponent_elo": args.opponent_elo,
        "games": args.games,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score": score,
        "elo_estimate": elo,
        "elo_ci95_low": elo_low,
        "elo_ci95_high": elo_high,
        "fm_target": args.fm_target,
        "status": "fm_level_confidence" if accepted else "not_yet_fm_level",
        "device": str(device),
        "git_sha": git_sha(Path(__file__).resolve().parents[2]),
    }
    write_report(args.report, report)
    if args.pgn:
        Path(args.pgn).parent.mkdir(parents=True, exist_ok=True)
        Path(args.pgn).write_text("\n\n".join(str(g) for g in pgn_games), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--pgn")
    parser.add_argument("--opponent-engine", default=os.environ.get("STOCKFISH_PATH"))
    parser.add_argument("--opponent-elo", type=float, default=2300.0)
    parser.add_argument("--fm-target", type=float, default=2300.0)
    parser.add_argument("--games", type=int, default=2)
    parser.add_argument("--max-ply", type=int, default=180)
    parser.add_argument("--movetime-ms", type=int, default=100)
    parser.add_argument("--nodes", type=int)
    parser.add_argument("--device", choices=["cpu", "cuda"])
    return parser


def main(argv: list[str] | None = None) -> int:
    play_ladder(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
