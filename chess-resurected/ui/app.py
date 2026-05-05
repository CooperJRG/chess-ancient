"""Flask server for Chess Resurected web UI."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import chess
import torch
from flask import Flask, Response, jsonify, request, send_from_directory

from engine_bridge import Engine
import training as tr

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "python"
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))

from chess_resurected.features import move_to_index
from chess_resurected.mcts import mcts_search
from chess_resurected.model import AlphaZeroNet

app = Flask(__name__, static_folder="static")
engine = Engine()
stockfish_engines: dict[int, Engine] = {}
custom_model: tuple[Path, AlphaZeroNet, torch.device] | None = None
play_opponent = "minimax"

STARTPOS = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

OPPONENTS = [
    {
        "id": "minimax",
        "label": "Basic Rust minimax",
        "kind": "Built-in",
        "description": "Depth-3 material search from the Rust engine.",
    },
    {
        "id": "stockfish-1",
        "label": "Stockfish level 1",
        "kind": "Stockfish",
        "description": "Very low Stockfish skill, useful for quick sanity games.",
    },
    {
        "id": "stockfish-5",
        "label": "Stockfish level 5",
        "kind": "Stockfish",
        "description": "Intermediate Stockfish setting.",
    },
    {
        "id": "stockfish-10",
        "label": "Stockfish level 10",
        "kind": "Stockfish",
        "description": "Strong Stockfish setting.",
    },
    {
        "id": "stockfish-20",
        "label": "Stockfish level 20",
        "kind": "Stockfish",
        "description": "Full Stockfish strength at the selected movetime.",
    },
    {
        "id": "custom",
        "label": "Trained custom model",
        "kind": "Neural",
        "description": "Latest checkpoint with MCTS search at move time.",
    },
]


@app.get("/")
def index():
    return send_from_directory("static", "index.html")


@app.get("/favicon.ico")
def favicon():
    return ("", 204)


def _stockfish_path() -> str:
    settings = tr.get_settings()
    return (
        settings.get("stockfish_path")
        or os.environ.get("STOCKFISH_PATH", "")
        or shutil.which("stockfish")
        or ""
    )


def _checkpoint_path() -> Path | None:
    run_name = tr.get_settings().get("run_name", "cuda-mainline")
    candidates = [
        ROOT / "models" / run_name / "latest.pt",
        ROOT / "models" / run_name / "baseline.pt",
        ROOT / "models" / "cuda-mainline" / "latest.pt",
        ROOT / "models" / "cuda-mainline" / "baseline.pt",
        ROOT / "models" / "baseline.pt",
        ROOT / "models" / "ui-smoke" / "baseline.pt",
    ]
    return next((path for path in candidates if path.exists()), None)


def _opponent_snapshot() -> dict:
    current = next((o for o in OPPONENTS if o["id"] == play_opponent), OPPONENTS[0])
    checkpoint = _checkpoint_path()
    return {
        "current": play_opponent,
        "current_label": current["label"],
        "stockfish_available": bool(_stockfish_path()),
        "custom_available": checkpoint is not None,
        "custom_checkpoint": str(checkpoint or ""),
        "opponents": OPPONENTS,
    }


def _get_stockfish(skill: int) -> Engine:
    path = _stockfish_path()
    if not path:
        raise RuntimeError("Stockfish is not configured")
    cached = stockfish_engines.get(skill)
    if cached is None:
        cached = Engine(path)
        cached.set_option("Skill Level", skill)
        stockfish_engines[skill] = cached
    return cached


def _load_custom_model() -> tuple[AlphaZeroNet, torch.device]:
    global custom_model
    path = _checkpoint_path()
    if path is None:
        raise RuntimeError("No trained checkpoint found")
    if custom_model is not None and custom_model[0] == path:
        return custom_model[1], custom_model[2]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    net = AlphaZeroNet(
        num_blocks=int(cfg.get("num_blocks", 4)),
        channels=int(cfg.get("channels", 64)),
        dropout_p=float(cfg.get("dropout_p", 0.1)),
    ).to(device)
    net.load_state_dict(ckpt["model"])
    net.eval()
    custom_model = (path, net, device)
    return net, device


def _checkmates_after(board: chess.Board, move: chess.Move) -> bool:
    child = board.copy(stack=False)
    child.push(move)
    return child.is_checkmate()


def _allows_immediate_mate(board: chess.Board, move: chess.Move) -> bool:
    child = board.copy(stack=False)
    child.push(move)
    return any(_checkmates_after(child, reply) for reply in child.legal_moves)


def _custom_best_move(fen: str) -> str | None:
    board = chess.Board(fen)
    legal = list(board.legal_moves)
    if not legal:
        return None
    for move in legal:
        if _checkmates_after(board, move):
            return move.uci()

    net, device = _load_custom_model()
    sims = int(tr.get_settings().get("interactive_simulations", 64))
    policy = mcts_search(board, net, device, n_simulations=max(1, sims), add_noise=False)
    ranked = sorted(legal, key=lambda m: float(policy[move_to_index(m, board)]), reverse=True)
    safe = [move for move in ranked if not _allows_immediate_mate(board, move)]
    return (safe[0] if safe else ranked[0]).uci()


@app.get("/play/opponents")
def play_opponents():
    return jsonify(_opponent_snapshot())


@app.post("/play/opponent")
def set_play_opponent():
    global play_opponent
    opponent_id = request.get_json(force=True).get("opponent")
    if opponent_id not in {o["id"] for o in OPPONENTS}:
        return jsonify({"ok": False, "error": "unknown opponent"}), 400
    play_opponent = opponent_id
    return jsonify({"ok": True, **_opponent_snapshot()})


@app.post("/move")
def move():
    data = request.get_json(force=True)
    current_fen = data.get("fen", STARTPOS)
    opponent = data.get("opponent") or play_opponent
    try:
        if opponent == "custom":
            engine_move = _custom_best_move(current_fen)
        elif opponent.startswith("stockfish-"):
            skill = int(opponent.split("-", 1)[1])
            engine_move = _get_stockfish(skill).get_best_move(current_fen, movetime_ms=700)
        else:
            engine_move = engine.get_best_move(current_fen, movetime_ms=800)
    except Exception as exc:
        return jsonify({"engine_move": None, "game_over": False, "error": str(exc), **_opponent_snapshot()}), 400

    if engine_move is None:
        return jsonify({"engine_move": None, "game_over": True, **_opponent_snapshot()})
    current = next((o for o in OPPONENTS if o["id"] == opponent), OPPONENTS[0])
    return jsonify({"engine_move": engine_move, "opponent": opponent, "opponent_label": current["label"]})


@app.post("/training/start")
def training_start():
    tr.start_training()
    return jsonify({"ok": True})


@app.post("/training/reset")
def training_reset():
    tr.reset_state()
    return jsonify({"ok": True})


@app.post("/training/stop")
def training_stop():
    tr.stop_training()
    return jsonify({"ok": True})


@app.get("/training/state")
def training_state():
    return jsonify(tr.get_state())


@app.get("/training/settings")
def get_settings_route():
    return jsonify(tr.get_settings())


@app.post("/training/settings")
def update_settings_route():
    tr.update_settings(request.get_json(force=True))
    return jsonify({"ok": True})


@app.get("/training/stream")
def training_stream():
    return Response(
        tr.event_stream(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    print("Chess Resurected UI at http://127.0.0.1:5000")
    app.run(debug=False, port=5000, threaded=True)
