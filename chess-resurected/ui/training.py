"""Pipeline runner for the Chess Resurected web UI."""

from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator

import torch
import chess

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_PKG = _ROOT / "python"
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from engine_bridge import Engine as UciEngine

from chess_resurected.filter_dataset import filter_records
from chess_resurected.dataset import read_jsonl, write_jsonl, write_report
from chess_resurected.engine_bridge import LegalMoveOracle
from chess_resurected.eval_baseline import evaluate
from chess_resurected.generate_selfplay import generate_selfplay_records
from chess_resurected.mcts_pipeline import (
    evaluate_replay,
    generate_mcts_cycle,
    load_replay,
    merge_replay,
    save_replay,
    train_replay,
)
from chess_resurected.model import AlphaZeroNet
from chess_resurected.play_ladder import play_ladder
from chess_resurected.train_baseline import train
from chess_resurected.dataset import stable_position_id, assign_split
from chess_resurected.records import PositionRecord

MODELS_DIR = _ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
SETTINGS_F = MODELS_DIR / "pipeline_settings.json"


def _engine_path() -> str:
    exe = "chess-resurected.exe" if os.name == "nt" else "chess-resurected"
    release = _ROOT / "engine" / "target" / "release" / exe
    debug = _ROOT / "engine" / "target" / "debug" / exe
    return str(release if release.exists() else debug)


def _stockfish_path() -> str:
    return os.environ.get("STOCKFISH_PATH", "") or shutil.which("stockfish") or ""


DEFAULT_SETTINGS: dict = {
    "run_name": "cuda-mainline",
    "preset": "balanced",
    "engine_path": _engine_path(),
    "stockfish_path": _stockfish_path(),
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "amp": True,
    "seed": 1,
    "pipeline_mode": "finite",
    "games": 1000,
    "cycle_games": 100,
    "max_cycles": 1,
    "warmup_cycles": 1,
    "max_ply": 140,
    "selfplay_mode": "mcts",
    "selfplay_movetime_ms": 30,
    "stockfish_nodes": 100,
    "mcts_simulations": 64,
    "interactive_simulations": 64,
    "random_opening_moves": 4,
    "temperature": 1.1,
    "temp_threshold": 24,
    "c_puct": 1.0,
    "draw_rate_warning": 0.75,
    "decisive_sample_weight": 2.0,
    "replay_buffer_size": 100000,
    "data_workers": 0,
    "train_epochs": 2,
    "batch_size": 512,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "num_blocks": 6,
    "channels": 128,
    "dropout_p": 0.1,
    "run_ladder": True,
    "ladder_games": 20,
    "ladder_movetime_ms": 100,
    "opponent_elo": 2300,
    "fm_target": 2300,
}

PRESETS = {
    "smoke": {"games": 10, "cycle_games": 2, "max_cycles": 1, "max_ply": 40, "mcts_simulations": 8, "train_epochs": 1, "batch_size": 64, "num_blocks": 2, "channels": 32, "ladder_games": 2},
    "balanced": {"games": 1000, "cycle_games": 100, "max_cycles": 1, "max_ply": 140, "mcts_simulations": 64, "train_epochs": 2, "batch_size": 512, "num_blocks": 6, "channels": 128, "ladder_games": 20},
    "serious": {"games": 10000, "cycle_games": 500, "max_cycles": 0, "max_ply": 180, "mcts_simulations": 200, "stockfish_nodes": 250, "train_epochs": 2, "batch_size": 512, "num_blocks": 8, "channels": 192, "ladder_games": 100},
}

_settings = dict(DEFAULT_SETTINGS)


@dataclass
class PipelineState:
    running: bool = False
    stage: str = "idle"
    status_msg: str = "Ready"
    progress: float = 0.0
    device_name: str = "cuda" if torch.cuda.is_available() else "cpu"
    cuda_available: bool = torch.cuda.is_available()
    started_at: float | None = None
    finished_at: float | None = None
    elapsed_sec: float = 0.0
    stage_current: int = 0
    stage_total: int = 0
    stage_detail: str = ""
    stage_rate: float = 0.0
    last_update_at: float | None = None
    log: list = field(default_factory=list)
    cycle: int = 0
    max_cycles: int = 0
    cycle_games: int = 0
    mcts_simulations: int = 0
    replay_samples: int = 0
    draw_rate: float = 0.0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    avg_ply: float = 0.0
    last_checkpoint_at: float | None = None
    run_name: str = ""
    generation_seconds: float = 0.0
    replay_load_seconds: float = 0.0
    replay_save_seconds: float = 0.0
    train_seconds_per_epoch: list = field(default_factory=list)
    checkpoint_write_seconds: list = field(default_factory=list)
    evaluation_seconds: float = 0.0
    ladder_seconds: float = 0.0
    event_push_ms: float = 0.0
    records_raw: int = 0
    records_filtered: int = 0
    train_loss: float = 0.0
    policy_loss: float = 0.0
    value_loss: float = 0.0
    policy_top1: float = 0.0
    policy_top5: float = 0.0
    value_mse: float = 0.0
    latency_ms: float = 0.0
    ladder_status: str = "not_run"
    elo_estimate: float = 0.0
    elo_low: float = 0.0
    elo_high: float = 0.0
    artifacts: dict = field(default_factory=dict)
    history: list = field(default_factory=list)
    error: str | None = None


state = PipelineState()
_q: queue.Queue[dict] = queue.Queue(maxsize=512)
_lock = threading.Lock()
_thread: threading.Thread | None = None
_stop_event = threading.Event()


def load_settings() -> None:
    global _settings
    if SETTINGS_F.exists():
        try:
            saved = json.loads(SETTINGS_F.read_text(encoding="utf-8"))
            _settings = {**DEFAULT_SETTINGS, **saved}
        except Exception:
            _settings = dict(DEFAULT_SETTINGS)


def save_settings() -> None:
    SETTINGS_F.write_text(json.dumps(_settings, indent=2), encoding="utf-8")


def get_settings() -> dict:
    return dict(_settings)


def update_settings(new: dict) -> None:
    preset = new.get("preset")
    if preset in PRESETS:
        _settings.update(PRESETS[preset])
    for key, value in new.items():
        if key in DEFAULT_SETTINGS:
            _settings[key] = value
    if not _settings.get("stockfish_path"):
        _settings["stockfish_path"] = _stockfish_path()
    save_settings()


load_settings()
if not _settings.get("stockfish_path"):
    _settings["stockfish_path"] = _stockfish_path()


def get_state() -> dict:
    with _lock:
        snap = asdict(state)
    if snap["running"] and snap["started_at"]:
        snap["elapsed_sec"] = time.time() - snap["started_at"]
    return snap


def reset_state() -> None:
    global state
    with _lock:
        state = PipelineState(
            device_name=str(_settings.get("device", DEFAULT_SETTINGS["device"])),
            run_name=str(_settings.get("run_name", "")),
        )
    _push({"status_msg": "Ready", "stage": "idle"})


def start_training() -> None:
    global _thread
    with _lock:
        if state.running:
            return
        state.running = True
        state.run_name = str(_settings.get("run_name", ""))
        state.stage = "queued"
        state.status_msg = "Pipeline queued"
        state.progress = 0.0
        state.stage_current = 0
        state.stage_total = 0
        state.stage_detail = ""
        state.stage_rate = 0.0
        state.cycle = 0
        state.max_cycles = int(_settings.get("max_cycles", 0))
        state.cycle_games = int(_settings.get("cycle_games", _settings.get("games", 0)))
        state.mcts_simulations = int(_settings.get("mcts_simulations", 0))
        state.replay_samples = 0
        state.draw_rate = 0.0
        state.wins = 0
        state.draws = 0
        state.losses = 0
        state.avg_ply = 0.0
        state.last_checkpoint_at = None
        state.generation_seconds = 0.0
        state.replay_load_seconds = 0.0
        state.replay_save_seconds = 0.0
        state.train_seconds_per_epoch = []
        state.checkpoint_write_seconds = []
        state.evaluation_seconds = 0.0
        state.ladder_seconds = 0.0
        state.event_push_ms = 0.0
        state.records_raw = 0
        state.records_filtered = 0
        state.train_loss = 0.0
        state.policy_loss = 0.0
        state.value_loss = 0.0
        state.policy_top1 = 0.0
        state.policy_top5 = 0.0
        state.value_mse = 0.0
        state.latency_ms = 0.0
        state.ladder_status = "not_run"
        state.elo_estimate = 0.0
        state.elo_low = 0.0
        state.elo_high = 0.0
        state.started_at = time.time()
        state.finished_at = None
        state.error = None
        state.history = []
        state.log = []
        state.device_name = str(_settings["device"])
    _stop_event.clear()
    _thread = threading.Thread(target=_run_pipeline, daemon=True)
    _thread.start()
    _push({})


def stop_training() -> None:
    _stop_event.set()
    _push({"status_msg": "Stopping after current stage", "stage": "stopping"})


def event_stream() -> Iterator[str]:
    yield _sse(get_state())
    while True:
        try:
            yield _sse(_q.get(timeout=15))
        except queue.Empty:
            yield ": heartbeat\n\n"


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


def _push(partial: dict) -> None:
    push_started = time.perf_counter()
    with _lock:
        now = time.time()
        for key, value in partial.items():
            if hasattr(state, key):
                setattr(state, key, value)
        state.event_push_ms = (time.perf_counter() - push_started) * 1000.0
        state.last_update_at = now
        if state.running and state.started_at:
            state.elapsed_sec = now - state.started_at
        snap = asdict(state)
    try:
        _q.put_nowait(snap)
    except queue.Full:
        pass


def _artifact_paths(run_name: str) -> dict[str, Path]:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in run_name).strip("-") or "run"
    run_dir = MODELS_DIR / safe
    run_dir.mkdir(parents=True, exist_ok=True)
    return {
        "run_dir": run_dir,
        "cycles_dir": run_dir / "cycles",
        "latest_checkpoint": run_dir / "latest.pt",
        "latest_replay": run_dir / "latest_replay.pt",
        "raw": run_dir / "selfplay_raw.jsonl",
        "raw_report": run_dir / "selfplay_raw_report.json",
        "filtered": run_dir / "selfplay_filtered.jsonl",
        "filter_report": run_dir / "selfplay_filter_report.json",
        "checkpoint": run_dir / "baseline.pt",
        "train_metrics": run_dir / "train_metrics.json",
        "train_config": run_dir / "train_config.json",
        "eval_report": run_dir / "eval_report.json",
        "ladder_report": run_dir / "ladder_report.json",
        "ladder_pgn": run_dir / "ladder_games.pgn",
    }


def _stop_requested() -> bool:
    return _stop_event.is_set()


def _stage(name: str, message: str, progress: float) -> None:
    _push({"stage": name, "status_msg": message, "progress": progress, "stage_current": 0, "stage_total": 0, "stage_detail": "", "stage_rate": 0.0})


def _log(message: str) -> None:
    with _lock:
        entries = list(state.log)
    entries.append({"t": time.strftime("%H:%M:%S"), "message": message})
    _push({"log": entries[-80:]})


def _stage_progress(
    *,
    current: int,
    total: int,
    detail: str,
    base: float,
    span: float,
    started: float,
    extra: dict | None = None,
) -> None:
    now = time.time()
    frac = current / total if total else 0.0
    payload = {
        "stage_current": current,
        "stage_total": total,
        "stage_detail": detail,
        "stage_rate": current / max(now - started, 0.001),
        "progress": min(base + span * frac, base + span),
        "status_msg": detail,
    }
    if extra:
        payload.update(extra)
    _push(payload)


def _result_for_color(result: str, color: chess.Color) -> float:
    if result == "1-0":
        return 1.0 if color == chess.WHITE else -1.0
    if result == "0-1":
        return 1.0 if color == chess.BLACK else -1.0
    return 0.0


def _generate_selfplay_records_streaming(settings: dict, oracle: LegalMoveOracle) -> list[PositionRecord]:
    import random

    rng = random.Random(int(settings["seed"]))
    records: list[PositionRecord] = []
    games = int(settings["games"])
    max_ply = int(settings["max_ply"])
    started = time.time()
    for game_idx in range(games):
        if _stop_requested():
            raise InterruptedError("Stopped during data generation")
        board = chess.Board()
        game_id = f"selfplay-random-seed{int(settings['seed'])}-game{game_idx:06d}"
        pending: list[tuple[str, str, tuple[str, ...], chess.Color, int]] = []
        for ply in range(max_ply):
            if board.is_game_over(claim_draw=True):
                break
            fen = board.fen()
            legal_moves = oracle.legal_moves(fen)
            legal = list(board.legal_moves)
            if not legal:
                break
            move = rng.choice(legal)
            pending.append((fen, move.uci(), legal_moves, board.turn, ply))
            board.push(move)
        result = board.result(claim_draw=True)
        split = assign_split(game_id)
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
                    split=split,
                    metadata={"generator": "selfplay", "mode": "random", "game_result": result},
                )
            )
        done = game_idx + 1
        if done == 1 or done % max(1, games // 100) == 0 or done == games:
            _stage_progress(
                current=done,
                total=games,
                detail=f"Generating self-play: game {done}/{games}, {len(records):,} records",
                base=0.08,
                span=0.20,
                started=started,
                extra={"records_raw": len(records)},
            )
            if done == 1 or done % max(1, games // 10) == 0 or done == games:
                _log(f"Generated {done}/{games} games ({len(records):,} records)")
    return records


def _cycle_paths(paths: dict[str, Path], cycle: int) -> dict[str, Path]:
    cycle_dir = paths["cycles_dir"] / f"{cycle:05d}"
    cycle_dir.mkdir(parents=True, exist_ok=True)
    return {
        "cycle_dir": cycle_dir,
        "replay": cycle_dir / "replay.pt",
        "report": cycle_dir / "cycle_report.json",
        "checkpoint": cycle_dir / "checkpoint.pt",
        "train_metrics": cycle_dir / "train_metrics.json",
        "train_config": cycle_dir / "train_config.json",
        "eval_report": cycle_dir / "eval_report.json",
        "ladder_report": cycle_dir / "ladder_report.json",
        "ladder_pgn": cycle_dir / "ladder_games.pgn",
    }


def _new_model(settings: dict) -> tuple[AlphaZeroNet, torch.device]:
    device = torch.device(settings.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    net = AlphaZeroNet(
        int(settings.get("num_blocks", 6)),
        int(settings.get("channels", 128)),
        float(settings.get("dropout_p", 0.1)),
    ).to(device)
    return net, device


def _load_latest_model(settings: dict, latest_checkpoint: Path) -> tuple[AlphaZeroNet, torch.device]:
    net, device = _new_model(settings)
    if latest_checkpoint.exists():
        ckpt = torch.load(latest_checkpoint, map_location=device, weights_only=False)
        cfg = ckpt.get("config", {})
        cfg_matches = (
            int(cfg.get("num_blocks", settings.get("num_blocks", 6))) == int(settings.get("num_blocks", 6))
            and int(cfg.get("channels", settings.get("channels", 128))) == int(settings.get("channels", 128))
            and float(cfg.get("dropout_p", settings.get("dropout_p", 0.1))) == float(settings.get("dropout_p", 0.1))
        )
        if cfg_matches:
            net.load_state_dict(ckpt["model"])
    net.eval()
    return net, device


def _run_mcts_pipeline(settings: dict, paths: dict[str, Path]) -> None:
    max_cycles = int(settings.get("max_cycles", 1))
    infinite = str(settings.get("pipeline_mode", "finite")) == "infinite" or max_cycles <= 0
    cycle = 1
    mode = str(settings.get("selfplay_mode", "mcts"))
    engine_path = str(settings["stockfish_path"]) if mode == "stockfish-copy" else str(settings["engine_path"])
    if mode == "stockfish-copy" and not engine_path.strip():
        raise RuntimeError("Stockfish-copy mode requires a Stockfish path")
    engine = UciEngine(engine_path)
    try:
        replay_load_started = time.perf_counter()
        existing_samples, existing_meta = load_replay(paths["latest_replay"])
        replay_load_seconds = time.perf_counter() - replay_load_started
        _push({"replay_samples": len(existing_samples), "replay_load_seconds": replay_load_seconds})
        if replay_load_seconds > 0.25:
            _log(f"Loaded replay buffer in {replay_load_seconds:.2f}s ({len(existing_samples):,} samples)")
        _log(
            f"Started MCTS {'infinite' if infinite else 'finite'} run '{settings['run_name']}' "
            f"on {settings['device']} with {settings['cycle_games']} games/cycle, source {mode}"
        )

        while not _stop_requested() and (infinite or cycle <= max_cycles):
            cycle_paths = _cycle_paths(paths, cycle)
            _push({
                "cycle": cycle,
                "max_cycles": max_cycles,
                "cycle_games": int(settings.get("cycle_games", settings.get("games", 1))),
                "mcts_simulations": int(settings.get("mcts_simulations", 64)),
            })
            _stage("generate", f"Cycle {cycle}: generating MCTS self-play", 0.05)
            _log(f"Cycle {cycle} started")
            net, device = _load_latest_model(settings, paths["latest_checkpoint"])
            generation_start = time.time()

            def _gen_progress(report: dict) -> None:
                wins = int(report.get("white_wins", 0)) + int(report.get("black_wins", 0))
                draws = int(report.get("draws", 0))
                losses = int(report.get("adjudicated_games", 0))
                draw_rate = float(report.get("draw_rate", 0.0))
                detail = (
                    f"Cycle {cycle}: game {report['game_current']}/{report['game_total']}, "
                    f"{report['samples']:,} samples, draws {draw_rate * 100:.1f}%"
                )
                _stage_progress(
                    current=int(report["game_current"]),
                    total=int(report["game_total"]),
                    detail=detail,
                    base=0.05,
                    span=0.30,
                    started=generation_start,
                    extra={
                        "records_raw": int(report["samples"]),
                        "wins": wins,
                        "draws": draws,
                        "losses": losses,
                        "draw_rate": draw_rate,
                        "avg_ply": float(report.get("average_ply", 0.0)),
                        "replay_samples": len(existing_samples) + int(report["samples"]),
                    },
                )

            cycle_samples, cycle_report = generate_mcts_cycle(
                cycle=cycle,
                engine=engine,
                net=net,
                device=device,
                settings=settings,
                stop_cb=_stop_requested,
                progress_cb=_gen_progress,
            )
            _push({
                "generation_seconds": float(cycle_report.get("seconds_total", 0.0)),
                "stage_rate": float(cycle_report.get("samples_per_second", 0.0)),
            })
            replay_save_started = time.perf_counter()
            save_replay(cycle_paths["replay"], cycle_samples, cycle_report)
            cycle_replay_save_seconds = time.perf_counter() - replay_save_started
            cycle_report["replay_save_seconds"] = cycle_replay_save_seconds
            write_report(cycle_paths["report"], cycle_report)
            if cycle_report["draw_rate"] >= float(settings.get("draw_rate_warning", 0.75)):
                _log(f"Cycle {cycle} draw-rate warning: {cycle_report['draw_rate'] * 100:.1f}%")
            _log(
                f"Cycle {cycle} generated {len(cycle_samples):,} samples "
                f"({cycle_report['white_wins']}W/{cycle_report['draws']}D/{cycle_report['black_wins']}L) "
                f"in {cycle_report.get('seconds_total', 0.0):.1f}s "
                f"({cycle_report.get('samples_per_second', 0.0):.1f} samples/s)"
            )

            existing_samples = merge_replay(
                existing_samples,
                cycle_samples,
                max_samples=int(settings.get("replay_buffer_size", 100000)),
                seed=int(settings.get("seed", 1)) + cycle,
            )
            replay_meta = {
                **existing_meta,
                "latest_cycle": cycle,
                "samples": len(existing_samples),
                "updated_at": time.time(),
                "last_cycle_report": cycle_report,
            }
            replay_save_started = time.perf_counter()
            save_replay(paths["latest_replay"], existing_samples, replay_meta)
            latest_replay_save_seconds = time.perf_counter() - replay_save_started
            _push({
                "replay_samples": len(existing_samples),
                "records_filtered": len(existing_samples),
                "replay_save_seconds": cycle_replay_save_seconds + latest_replay_save_seconds,
            })
            if latest_replay_save_seconds > 0.25:
                _log(f"Saved latest replay in {latest_replay_save_seconds:.2f}s ({len(existing_samples):,} samples)")
            if _stop_requested():
                _log("Stop requested after generation; replay artifacts were written")
                break

            _stage("train", f"Cycle {cycle}: training from MCTS replay", 0.38)
            train_start = time.time()

            def _train_progress(metrics: dict) -> None:
                _stage_progress(
                    current=int(metrics["epoch"]),
                    total=int(metrics["epochs"]),
                    detail=f"Cycle {cycle}: epoch {metrics['epoch']}/{metrics['epochs']}, loss {metrics['total_loss']:.4f}",
                    base=0.38,
                    span=0.34,
                    started=train_start,
                    extra={
                        "train_loss": float(metrics.get("total_loss", 0.0)),
                        "policy_loss": float(metrics.get("policy_loss", 0.0)),
                        "value_loss": float(metrics.get("value_loss", 0.0)),
                    },
                )
                _log(
                    f"Cycle {cycle} epoch {metrics['epoch']}: loss {metrics['total_loss']:.4f}, "
                    f"policy {metrics['policy_loss']:.4f}, value {metrics['value_loss']:.4f}"
                )

            train_report = train_replay(
                replay_path=paths["latest_replay"],
                checkpoint_path=cycle_paths["checkpoint"],
                metrics_path=cycle_paths["train_metrics"],
                config_path=cycle_paths["train_config"],
                settings=settings,
                samples=existing_samples,
                replay_metadata=replay_meta,
                progress_cb=_train_progress,
            )
            last = train_report["history"][-1] if train_report.get("history") else {}
            shutil.copy2(cycle_paths["checkpoint"], paths["latest_checkpoint"])
            shutil.copy2(cycle_paths["checkpoint"], paths["checkpoint"])
            _push({
                "train_loss": float(last.get("total_loss", 0.0)),
                "policy_loss": float(last.get("policy_loss", 0.0)),
                "value_loss": float(last.get("value_loss", 0.0)),
                "last_checkpoint_at": time.time(),
                "replay_load_seconds": float(train_report.get("replay_load_seconds", 0.0)),
                "train_seconds_per_epoch": train_report.get("train_seconds_per_epoch", []),
                "checkpoint_write_seconds": train_report.get("checkpoint_write_seconds", []),
            })
            _log(f"Cycle {cycle} checkpoint written: {cycle_paths['checkpoint']}")
            if train_report.get("train_seconds_per_epoch"):
                _log(
                    f"Cycle {cycle} training timing: "
                    f"{sum(train_report['train_seconds_per_epoch']):.1f}s total, "
                    f"{train_report['train_seconds_per_epoch'][-1]:.1f}s last epoch"
                )
            if _stop_requested():
                _log("Stop requested after training; checkpoint is complete")
                break

            _stage("evaluate", f"Cycle {cycle}: evaluating latest checkpoint", 0.75)
            eval_started = time.perf_counter()
            eval_report = evaluate_replay(
                replay_path=paths["latest_replay"],
                checkpoint_path=paths["latest_checkpoint"],
                report_path=cycle_paths["eval_report"],
                settings=settings,
                samples=existing_samples,
            )
            evaluation_seconds = time.perf_counter() - eval_started
            shutil.copy2(cycle_paths["eval_report"], paths["eval_report"])
            _push({
                "policy_top1": float(eval_report["policy_top1"]),
                "policy_top5": float(eval_report["policy_top5"]),
                "value_mse": float(eval_report["value_mse"]),
                "latency_ms": float(eval_report["latency_ms_per_position"]),
                "evaluation_seconds": evaluation_seconds,
                "stage_current": int(eval_report["records"]),
                "stage_total": int(eval_report["records"]),
                "stage_detail": f"Cycle {cycle}: evaluated {int(eval_report['records']):,} replay samples",
            })
            _log(
                f"Cycle {cycle} eval: top1 {eval_report['policy_top1'] * 100:.1f}%, "
                f"top5 {eval_report['policy_top5'] * 100:.1f}%, value MSE {eval_report['value_mse']:.5f}, "
                f"{evaluation_seconds:.1f}s"
            )

            if bool(settings["run_ladder"]) and not infinite and str(settings.get("stockfish_path", "")).strip():
                _stage("ladder", f"Cycle {cycle}: running Stockfish ladder", 0.88)
                ladder_started = time.perf_counter()
                ladder = play_ladder(
                    argparse.Namespace(
                        checkpoint=str(paths["latest_checkpoint"]),
                        report=str(cycle_paths["ladder_report"]),
                        pgn=str(cycle_paths["ladder_pgn"]),
                        opponent_engine=str(settings["stockfish_path"]),
                        opponent_elo=float(settings["opponent_elo"]),
                        fm_target=float(settings["fm_target"]),
                        games=int(settings["ladder_games"]),
                        max_ply=int(settings["max_ply"]),
                        movetime_ms=int(settings["ladder_movetime_ms"]),
                        nodes=None,
                        device=str(settings["device"]),
                    )
                )
                ladder_seconds = time.perf_counter() - ladder_started
                shutil.copy2(cycle_paths["ladder_report"], paths["ladder_report"])
                if cycle_paths["ladder_pgn"].exists():
                    shutil.copy2(cycle_paths["ladder_pgn"], paths["ladder_pgn"])
                _push({
                    "ladder_status": str(ladder["status"]),
                    "elo_estimate": float(ladder["elo_estimate"]),
                    "elo_low": float(ladder["elo_ci95_low"]),
                    "elo_high": float(ladder["elo_ci95_high"]),
                    "ladder_seconds": ladder_seconds,
                })
                _log(f"Cycle {cycle} ladder: Elo {ladder['elo_estimate']:.0f}, {ladder['status']}")
            elif infinite:
                _push({"ladder_status": "skipped_in_infinite_mode"})

            _stage("checkpoint", f"Cycle {cycle}: checkpoint durable", min(0.99, 0.90 + 0.05 * (cycle / max(max_cycles, 1))))
            _log(f"Cycle {cycle} complete")
            cycle += 1

        if _stop_requested():
            _push({"stage": "stopped", "status_msg": "Stopped cleanly after current cycle stage", "error": None})
            _log("Pipeline stopped cleanly")
        else:
            _stage("complete", "Pipeline complete", 1.0)
            _log("Pipeline complete")
    finally:
        engine.close()


def _run_pipeline() -> None:
    settings = get_settings()
    paths = _artifact_paths(str(settings["run_name"]))
    _push({"artifacts": {k: str(v) for k, v in paths.items()}})
    try:
        if str(settings.get("selfplay_mode", "mcts")) in {"mcts", "stockfish-copy", "rust-copy"}:
            _run_mcts_pipeline(settings, paths)
            return

        oracle = LegalMoveOracle(settings["engine_path"])

        _stage("generate", "Generating self-play data", 0.08)
        _log(f"Started run '{settings['run_name']}' on {settings['device']} with {settings['games']} games")
        if str(settings["selfplay_mode"]) == "random":
            records = _generate_selfplay_records_streaming(settings, oracle)
        else:
            _log("Engine-mode self-play reports progress after the generation stage completes")
            records = generate_selfplay_records(
                games=int(settings["games"]),
                max_ply=int(settings["max_ply"]),
                seed=int(settings["seed"]),
                oracle=oracle,
                mode=str(settings["selfplay_mode"]),
                engine_path=str(settings["engine_path"]),
                movetime_ms=int(settings["selfplay_movetime_ms"]),
            )
        write_jsonl(paths["raw"], records)
        write_report(paths["raw_report"], {"records": len(records), "settings": settings})
        _push({"records_raw": len(records), "history": state.history + [["generated", len(records)]]})
        _log(f"Wrote raw dataset with {len(records):,} records")
        if _stop_requested():
            raise InterruptedError("Stopped after data generation")

        _stage("filter", "Filtering illegal and duplicate records", 0.28)
        _log("Filtering illegal moves and duplicate positions")
        kept, report = filter_records(records, oracle=oracle)
        write_jsonl(paths["filtered"], kept)
        write_report(paths["filter_report"], report)
        _push({"records_filtered": len(kept), "history": state.history + [["filtered", len(kept)]], "stage_current": len(records), "stage_total": len(records), "stage_detail": f"Filtered {len(kept):,}/{len(records):,} records"})
        _log(f"Filtered dataset: {len(kept):,} kept, {report.get('removed_records', 0):,} removed")
        if _stop_requested():
            raise InterruptedError("Stopped after filtering")

        _stage("train", "Training baseline on CUDA" if settings["device"] == "cuda" else "Training baseline", 0.45)
        _log(f"Training {settings['train_epochs']} epoch(s), batch {settings['batch_size']}, blocks {settings['num_blocks']}, channels {settings['channels']}")
        train_start = time.time()
        _push({"stage_current": 0, "stage_total": int(settings["train_epochs"]), "stage_detail": "Training starting"})
        def _train_progress(done: int, total: int, metrics: dict) -> None:
            _stage_progress(
                current=done,
                total=total,
                detail=f"Training epoch {done}/{total}: loss {metrics.get('total_loss', 0):.4f}",
                base=0.45,
                span=0.27,
                started=train_start,
                extra={
                    "train_loss": float(metrics.get("total_loss", 0.0)),
                    "policy_loss": float(metrics.get("policy_loss", 0.0)),
                    "value_loss": float(metrics.get("value_loss", 0.0)),
                },
            )
            _log(f"Epoch {metrics.get('epoch')}: loss {metrics.get('total_loss'):.4f}, policy {metrics.get('policy_loss'):.4f}, value {metrics.get('value_loss'):.4f}")
        train_report = train(
            argparse.Namespace(
                dataset=str(paths["filtered"]),
                checkpoint=str(paths["checkpoint"]),
                metrics=str(paths["train_metrics"]),
                config_out=str(paths["train_config"]),
                epochs=int(settings["train_epochs"]),
                batch_size=int(settings["batch_size"]),
                lr=float(settings["learning_rate"]),
                weight_decay=float(settings["weight_decay"]),
                num_blocks=int(settings["num_blocks"]),
                channels=int(settings["channels"]),
                dropout_p=float(settings["dropout_p"]),
                seed=int(settings["seed"]),
                device=str(settings["device"]),
                resume=False,
                amp=bool(settings["amp"]),
            ),
            progress_cb=_train_progress,
        )
        last = train_report["history"][-1] if train_report.get("history") else {}
        _push({
            "train_loss": float(last.get("total_loss", 0.0)),
            "policy_loss": float(last.get("policy_loss", 0.0)),
            "value_loss": float(last.get("value_loss", 0.0)),
            "history": state.history + [["trained", float(last.get("total_loss", 0.0))]],
            "stage_current": int(settings["train_epochs"]),
            "stage_total": int(settings["train_epochs"]),
            "stage_detail": f"Training complete in {time.time() - train_start:.1f}s",
        })
        if _stop_requested():
            raise InterruptedError("Stopped after training")

        _stage("evaluate", "Evaluating checkpoint", 0.72)
        _log("Evaluating checkpoint on filtered dataset")
        eval_report = evaluate(
            argparse.Namespace(
                dataset=str(paths["filtered"]),
                checkpoint=str(paths["checkpoint"]),
                report=str(paths["eval_report"]),
                split=None,
                batch_size=int(settings["batch_size"]),
                top_k=5,
                device=str(settings["device"]),
            )
        )
        _push({
            "policy_top1": float(eval_report["policy_top1"]),
            "policy_top5": float(eval_report["policy_top5"]),
            "value_mse": float(eval_report["value_mse"]),
            "latency_ms": float(eval_report["latency_ms_per_position"]),
            "stage_current": int(eval_report["records"]),
            "stage_total": int(eval_report["records"]),
            "stage_detail": f"Evaluated {int(eval_report['records']):,} records",
        })
        _log(f"Evaluation: top1 {eval_report['policy_top1']*100:.1f}%, top5 {eval_report['policy_top5']*100:.1f}%, value MSE {eval_report['value_mse']:.5f}")

        if bool(settings["run_ladder"]) and str(settings.get("stockfish_path", "")).strip():
            _stage("ladder", "Running Stockfish ladder", 0.86)
            _log(f"Running ladder: {settings['ladder_games']} games vs Stockfish")
            ladder_start = time.time()
            _push({"stage_current": 0, "stage_total": int(settings["ladder_games"]), "stage_detail": f"Ladder starting: 0/{settings['ladder_games']} games"})
            def _ladder_progress(done: int, total: int, partial: dict) -> None:
                _stage_progress(
                    current=done,
                    total=total,
                    detail=f"Ladder game {done}/{total}: {partial['wins']}W/{partial['draws']}D/{partial['losses']}L",
                    base=0.86,
                    span=0.13,
                    started=ladder_start,
                )
                _log(f"Ladder {done}/{total}: {partial['wins']}W/{partial['draws']}D/{partial['losses']}L")
            ladder = play_ladder(
                argparse.Namespace(
                    checkpoint=str(paths["checkpoint"]),
                    report=str(paths["ladder_report"]),
                    pgn=str(paths["ladder_pgn"]),
                    opponent_engine=str(settings["stockfish_path"]),
                    opponent_elo=float(settings["opponent_elo"]),
                    fm_target=float(settings["fm_target"]),
                    games=int(settings["ladder_games"]),
                    max_ply=int(settings["max_ply"]),
                    movetime_ms=int(settings["ladder_movetime_ms"]),
                    nodes=None,
                    device=str(settings["device"]),
                ),
                progress_cb=_ladder_progress,
            )
            _push({
                "ladder_status": str(ladder["status"]),
                "elo_estimate": float(ladder["elo_estimate"]),
                "elo_low": float(ladder["elo_ci95_low"]),
                "elo_high": float(ladder["elo_ci95_high"]),
                "stage_current": int(settings["ladder_games"]),
                "stage_total": int(settings["ladder_games"]),
                "stage_detail": f"Ladder complete: {ladder['wins']}W/{ladder['draws']}D/{ladder['losses']}L",
            })
            _log(f"Ladder complete: Elo {ladder['elo_estimate']:.0f} [{ladder['elo_ci95_low']:.0f}, {ladder['elo_ci95_high']:.0f}], {ladder['status']}")
        else:
            _push({"ladder_status": "skipped_no_stockfish"})
            _log("Ladder skipped because Stockfish is not configured")

        _stage("complete", "Pipeline complete", 1.0)
        _log("Pipeline complete")
    except InterruptedError as exc:
        _push({"stage": "stopped", "status_msg": str(exc), "error": None})
        _log(str(exc))
    except Exception as exc:
        _push({"stage": "error", "status_msg": f"Pipeline failed: {exc}", "error": str(exc)})
        _log(f"Pipeline failed: {exc}")
    finally:
        _push({"running": False, "finished_at": time.time()})
