"""Self-play game generation with MCTS.

Two modes:
  engine_game  – both sides use the UCI engine (warm-start data).
                 Policy target is a one-hot on the move played (canonical idx).
  nn_game      – both sides use MCTS guided by the neural network.
                 Policy target is the full MCTS visit-count distribution.

Value targets are from the perspective of the side to move:
  +1 = that player's side won, -1 = lost, 0 = draw.

Each Sample carries a legal_mask so the training loss can be computed only
over legal moves (masked cross-entropy), preventing probability leakage onto
illegal moves.
"""
from __future__ import annotations

import random
import chess
import numpy as np
import torch
from typing import NamedTuple, Callable

from .features import board_to_tensor, move_to_index, NUM_POLICY
from .mcts import mcts_search


_PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}


def _material_score(board: chess.Board) -> int:
    score = 0
    for piece_type, value in _PIECE_VALUES.items():
        score += len(board.pieces(piece_type, chess.WHITE)) * value
        score -= len(board.pieces(piece_type, chess.BLACK)) * value
    return score


def _adjudicated_result(board: chess.Board, material_threshold: int = 3) -> str:
    result = board.result(claim_draw=True)
    if result != "*":
        return result
    score = _material_score(board)
    if score >= material_threshold:
        return "1-0"
    if score <= -material_threshold:
        return "0-1"
    return "1/2-1/2"


class Sample(NamedTuple):
    tensor:     torch.Tensor   # (18, 8, 8) float32
    policy:     torch.Tensor   # (NUM_POLICY,) float32  — MCTS visit-count distribution
    value:      float          # +1 / 0 / -1 from side-to-move perspective
    legal_mask: torch.Tensor   # (NUM_POLICY,) bool — True where move is legal


# ── helpers ────────────────────────────────────────────────────────────────────

def _outcome_to_values(
    board: chess.Board,
    history: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, chess.Color]],
) -> list[Sample]:
    result = _adjudicated_result(board)
    winner = (chess.WHITE if result == "1-0" else
              chess.BLACK if result == "0-1" else None)
    samples = []
    for tensor, policy, mask, color in history:
        val = 0.0 if winner is None else (1.0 if color == winner else -1.0)
        samples.append(Sample(tensor, policy, val, mask))
    return samples


def _random_opening(board: chess.Board, num_moves: int) -> None:
    for _ in range(num_moves):
        if board.is_game_over(claim_draw=True):
            break
        legal = list(board.legal_moves)
        if not legal:
            break
        board.push(random.choice(legal))


def _legal_mask(board: chess.Board) -> torch.Tensor:
    """Build a bool tensor of shape (NUM_POLICY,) marking canonical legal moves."""
    mask = torch.zeros(NUM_POLICY, dtype=torch.bool)
    for m in board.legal_moves:
        mask[move_to_index(m, board)] = True
    return mask


def _one_hot_policy(move: chess.Move, board: chess.Board) -> torch.Tensor:
    p = torch.zeros(NUM_POLICY, dtype=torch.float32)
    p[move_to_index(move, board)] = 1.0
    return p


# ── engine self-play ───────────────────────────────────────────────────────────

def engine_game(engine,
                movetime_ms: int = 150,
                random_opening_moves: int = 8,
                max_ply: int = 150) -> tuple[list[Sample], dict]:
    """Play one game with the UCI engine on both sides."""
    board = chess.Board()
    _random_opening(board, random_opening_moves)
    history: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, chess.Color]] = []

    for _ in range(max_ply):
        if board.is_game_over(claim_draw=True):
            break
        uci = engine.get_best_move(board.fen(), movetime_ms=movetime_ms)
        if uci is None:
            break
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            break
        t    = board_to_tensor(board)
        pol  = _one_hot_policy(move, board)
        mask = _legal_mask(board)
        history.append((t, pol, mask, board.turn))
        board.push(move)

    samples = _outcome_to_values(board, history)
    result = _adjudicated_result(board)
    return samples, {"length": len(history), "result": result}


# ── neural network self-play with MCTS ────────────────────────────────────────

def nn_game(net,
            device: torch.device,
            temperature: float = 1.0,
            random_opening_moves: int = 8,
            max_ply: int = 150,
            n_simulations: int = 100,
            c_puct: float = 1.0,
            dirichlet_alpha: float = 0.3,
            dirichlet_eps: float = 0.25,
            temp_threshold: int = 30) -> tuple[list[Sample], dict]:
    """Play one game using MCTS guided by the neural network."""
    net.eval()
    board = chess.Board()
    _random_opening(board, random_opening_moves)
    history: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, chess.Color]] = []

    for move_num in range(max_ply):
        if board.is_game_over(claim_draw=True):
            break
        legal = list(board.legal_moves)
        if not legal:
            break

        t    = board_to_tensor(board)
        mask = _legal_mask(board)

        pi = mcts_search(
            board, net, device,
            n_simulations=n_simulations,
            c_puct=c_puct,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_eps=dirichlet_eps,
            add_noise=True,
        )

        # Move selection — pi is already in canonical coordinates
        legal_idxs = [move_to_index(m, board) for m in legal]
        legal_pi   = np.array([pi[i] for i in legal_idxs], dtype=np.float32)

        if move_num < temp_threshold and temperature > 0:
            if temperature != 1.0:
                legal_pi = np.power(np.maximum(legal_pi, 1e-10), 1.0 / temperature)
            total = legal_pi.sum()
            legal_pi = legal_pi / total if total > 1e-9 else np.ones(len(legal)) / len(legal)
            chosen = int(np.random.choice(len(legal), p=legal_pi))
        else:
            chosen = int(np.argmax(legal_pi))

        move = legal[chosen]
        history.append((t, torch.from_numpy(pi), mask, board.turn))
        board.push(move)

    samples = _outcome_to_values(board, history)
    result = board.result(claim_draw=True)
    return samples, {"length": len(history), "result": result}


# ── batch generation ───────────────────────────────────────────────────────────

def generate_games(engine, net, device,
                   num_games: int,
                   settings: dict,
                   progress_cb: Callable | None = None,
                   epoch: int = 0) -> tuple[list[Sample], dict]:
    """Generate `num_games` games mixing engine and MCTS NN self-play."""
    warmup_epochs    = settings.get("warmup_epochs",         3)
    nn_pct           = settings.get("nn_self_play_pct",     80) / 100.0
    movetime_ms      = settings.get("movetime_ms",         150)
    temperature      = settings.get("temperature",         1.2)
    random_open      = settings.get("random_opening_moves",  8)
    max_ply          = settings.get("max_ply",             150)
    n_simulations    = settings.get("n_simulations",       200)
    c_puct           = settings.get("c_puct",              1.0)
    dirichlet_alpha  = settings.get("dirichlet_alpha",     0.3)
    dirichlet_eps    = settings.get("dirichlet_eps",      0.25)
    temp_threshold   = settings.get("temp_threshold",       30)

    # Anneal temperature: start high, decay each epoch after warmup
    temp = max(0.5, temperature * (0.95 ** max(0, epoch - warmup_epochs)))

    all_samples: list[Sample] = []
    results = {"W": 0, "D": 0, "L": 0, "total_ply": 0}

    for i in range(num_games):
        use_nn = (epoch >= warmup_epochs) and (random.random() < nn_pct) and (net is not None)

        if use_nn:
            samples, stats = nn_game(
                net, device,
                temperature          = temp,
                random_opening_moves = random_open,
                max_ply              = max_ply,
                n_simulations        = n_simulations,
                c_puct               = c_puct,
                dirichlet_alpha      = dirichlet_alpha,
                dirichlet_eps        = dirichlet_eps,
                temp_threshold       = temp_threshold,
            )
        else:
            samples, stats = engine_game(
                engine,
                movetime_ms          = movetime_ms,
                random_opening_moves = random_open,
                max_ply              = max_ply,
            )

        all_samples.extend(samples)
        results["total_ply"] += stats["length"]
        r = stats["result"]
        if r in ("1-0", "0-1"):
            results["W"] += 1
        elif r == "1/2-1/2":
            results["D"] += 1
        else:
            results["L"] += 1

        if progress_cb:
            progress_cb(i + 1, num_games, len(all_samples),
                        "nn" if use_nn else "engine", temp)

    avg_len = results["total_ply"] / max(num_games, 1)
    return all_samples, {
        **results,
        "avg_game_length": round(avg_len, 1),
        "temperature":     round(temp, 3),
    }
