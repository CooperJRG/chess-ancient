"""Flask server for Chess Resurected web UI."""

from __future__ import annotations

import json
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

_api_cache: dict[str, tuple[float, object]] = {}
_API_TTL = 30.0

def _cached_api(key: str, fn):
    import time as _time
    now = _time.time()
    if key in _api_cache and now - _api_cache[key][0] < _API_TTL:
        return _api_cache[key][1]
    result = fn()
    _api_cache[key] = (now, result)
    return result

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


@app.get("/run")
def run_page():
    return send_from_directory("static", "run.html")


@app.get("/play")
def play_page():
    return send_from_directory("static", "play.html")


@app.get("/ladder")
def ladder_page():
    return send_from_directory("static", "ladder.html")


@app.get("/registry")
def registry_page():
    return send_from_directory("static", "registry.html")


@app.get("/search")
def search_page():
    return send_from_directory("static", "search.html")


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


def _build_ladder():
    models_dir = ROOT / "models"
    results = []

    def _read_report(path: Path, run: str, cycle: int | None = None) -> dict | None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            data["run"] = run
            if cycle is not None:
                data["cycle"] = cycle
            return data
        except Exception:
            return None

    for run_dir in sorted(models_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        rn = run_dir.name
        cycles_dir = run_dir / "cycles"
        if cycles_dir.exists():
            for cycle_dir in sorted(cycles_dir.iterdir(), reverse=True):
                rpt = cycle_dir / "ladder_report.json"
                if rpt.exists():
                    entry = _read_report(rpt, rn, int(cycle_dir.name) if cycle_dir.name.isdigit() else None)
                    if entry:
                        results.append(entry)
        rpt = run_dir / "ladder_report.json"
        if rpt.exists():
            entry = _read_report(rpt, rn)
            if entry:
                results.append(entry)

    root_rpt = models_dir / "ladder_report.json"
    if root_rpt.exists():
        entry = _read_report(root_rpt, "baseline")
        if entry:
            results.append(entry)

    results.sort(key=lambda r: r.get("elo_estimate", 0), reverse=True)
    return jsonify({"runs": results, "total": len(results)})


@app.get("/api/ladder")
def api_ladder():
    return _cached_api("ladder", _build_ladder)


def _build_registry():
    models_dir = ROOT / "models"
    runs = []

    def _load_json(path: Path) -> dict:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    for run_dir in sorted(models_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        rn = run_dir.name
        checkpoints = []

        cycles_dir = run_dir / "cycles"
        if cycles_dir.exists():
            for cycle_dir in sorted(cycles_dir.iterdir()):
                if not cycle_dir.is_dir():
                    continue
                cycle_num = int(cycle_dir.name) if cycle_dir.name.isdigit() else None
                if not (cycle_dir / "checkpoint.pt").exists():
                    continue
                info: dict = {"name": f"{rn}-cycle-{cycle_dir.name}", "type": "cycle", "run": rn}
                if cycle_num is not None:
                    info["cycle"] = cycle_num
                m = _load_json(cycle_dir / "train_metrics.json")
                h = m.get("history", [])
                if h:
                    last = h[-1]
                    info["total_loss"] = last.get("total_loss")
                    info["policy_loss"] = last.get("policy_loss")
                    info["value_loss"] = last.get("value_loss")
                    info["epoch"] = last.get("epoch")
                ev = _load_json(cycle_dir / "eval_report.json")
                if ev:
                    info["policy_top1"] = ev.get("policy_top1")
                    info["policy_top5"] = ev.get("policy_top5")
                    info["value_mse"] = ev.get("value_mse")
                lad = _load_json(cycle_dir / "ladder_report.json")
                if lad:
                    info["elo"] = lad.get("elo_estimate")
                    info["wins"] = lad.get("wins")
                    info["losses"] = lad.get("losses")
                    info["draws"] = lad.get("draws")
                checkpoints.append(info)

        if (run_dir / "latest.pt").exists():
            info = {"name": f"{rn}-latest", "type": "latest", "run": rn}
            m = _load_json(run_dir / "train_metrics.json")
            h = m.get("history", [])
            if h:
                last = h[-1]
                info["total_loss"] = last.get("total_loss")
                info["policy_loss"] = last.get("policy_loss")
                info["value_loss"] = last.get("value_loss")
                info["epoch"] = last.get("epoch")
            ev = _load_json(run_dir / "eval_report.json")
            if ev:
                info["policy_top1"] = ev.get("policy_top1")
                info["value_mse"] = ev.get("value_mse")
            lad = _load_json(run_dir / "ladder_report.json")
            if lad:
                info["elo"] = lad.get("elo_estimate")
            checkpoints.append(info)

        if checkpoints:
            runs.append({"run": rn, "checkpoints": checkpoints})

    total = sum(len(r["checkpoints"]) for r in runs)
    return jsonify({"runs": runs, "total": total})


@app.get("/api/registry")
def api_registry():
    return _cached_api("registry", _build_registry)


@app.get("/api/hint")
def api_hint():
    """Run MCTS on a FEN and return top moves (used by play page for engine thinking)."""
    fen = request.args.get("fen", "")
    sims = min(int(request.args.get("sims", 32)), 256)
    if not fen:
        return jsonify({"moves": []})
    try:
        board = chess.Board(fen)
        if board.is_game_over(claim_draw=True):
            return jsonify({"moves": []})
        net, device = _load_custom_model()
        policy = mcts_search(board, net, device, n_simulations=sims, add_noise=False)
        legal = list(board.legal_moves)
        moves_out = []
        for move in sorted(legal, key=lambda m: float(policy[move_to_index(m, board)]), reverse=True)[:5]:
            idx = move_to_index(move, board)
            pr = float(policy[idx])
            # Quick value estimate via single forward pass
            with torch.no_grad():
                t = __import__("chess_resurected.features", fromlist=["board_to_tensor"]).board_to_tensor(board)
                v, _ = net(t.unsqueeze(0).to(device))
                q = float(v.squeeze())
            moves_out.append({
                "move": board.san(move),
                "uci": move.uci(),
                "visits": max(1, int(pr * sims)),
                "q": round(q, 3),
                "prior": round(pr, 4),
                "from": move.uci()[:2],
                "to": move.uci()[2:4],
            })
        return jsonify({"moves": moves_out})
    except Exception as exc:
        return jsonify({"moves": [], "error": str(exc)})


@app.get("/api/search")
def api_search():
    """Run MCTS and return a tree structure for the search visualiser."""
    fen = request.args.get("fen", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    sims = min(int(request.args.get("sims", 64)), 512)
    try:
        board = chess.Board(fen)
        if board.is_game_over(claim_draw=True):
            return jsonify({"error": "Game is already over."})
        net, device = _load_custom_model()
        policy = mcts_search(board, net, device, n_simulations=sims, add_noise=False)
        legal = list(board.legal_moves)
        top_moves = sorted(legal, key=lambda m: float(policy[move_to_index(m, board)]), reverse=True)[:8]

        # Build a simple tree representation for the SVG renderer
        nodes, edges, node_id = [], [], [0]
        def add_node(parent_id, depth, visits, q, x, y, move=None):
            nid = node_id[0]; node_id[0] += 1
            nodes.append({"id": nid, "depth": depth, "visits": visits, "q": q, "x": x, "y": y, "move": move})
            if parent_id is not None:
                edges.append([parent_id, nid])
            return nid

        # Root
        total_visits = sims
        root_id = add_node(None, 0, total_visits, 0.0, 500, 60)

        # Depth 1 — top moves spread across width
        n_top = len(top_moves)
        spacing = 900 / max(n_top, 1)
        d1_ids = []
        for i, move in enumerate(top_moves):
            idx = move_to_index(move, board)
            pr = float(policy[idx])
            vis = max(1, int(pr * total_visits))
            x = 50 + (i + 0.5) * spacing
            nid = add_node(root_id, 1, vis, 0.0, x, 220, board.san(move))
            d1_ids.append((nid, x, vis))

        # Depth 2 — a few children per top move
        for parent_id, px, pvis in d1_ids[:5]:
            n_children = max(1, min(4, int(pvis / (total_visits / 6))))
            for k in range(n_children):
                vis2 = max(1, int(pvis * (0.5 - k * 0.1)))
                x2 = px + (k - n_children / 2) * (spacing * 0.7 / max(n_children, 1))
                add_node(parent_id, 2, vis2, 0.0, x2, 400)

        moves_out = []
        for move in top_moves:
            idx = move_to_index(move, board)
            pr = float(policy[idx])
            vis = max(1, int(pr * total_visits))
            with torch.no_grad():
                from chess_resurected.features import board_to_tensor
                t = board_to_tensor(board)
                v, _ = net(t.unsqueeze(0).to(device))
                q = float(v.squeeze())
            moves_out.append({
                "move": board.san(move), "uci": move.uci(),
                "visits": vis, "q": round(q, 3), "prior": round(pr, 4),
                "from": move.uci()[:2], "to": move.uci()[2:4],
            })

        return jsonify({
            "tree": {"nodes": nodes, "edges": edges},
            "moves": moves_out,
            "total_visits": total_visits,
            "fen": fen,
        })
    except RuntimeError as exc:
        return jsonify({"error": str(exc)})
    except Exception as exc:
        return jsonify({"error": f"Search failed: {exc}"})


if __name__ == "__main__":
    print("Chess Resurected UI at http://127.0.0.1:5000")
    app.run(debug=False, port=5000, threaded=True)
