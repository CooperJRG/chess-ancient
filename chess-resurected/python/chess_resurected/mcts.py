"""PUCT-MCTS for AlphaZero self-play and evaluation.

Each call to mcts_search grows a search tree rooted at the current board
position, running n_simulations rollouts guided by the neural network.
Returns a normalized visit-count policy vector of shape (NUM_POLICY,).

Value convention: all node.value_sum values are stored from the perspective
of the player to move at that node's board position.  PUCT selection negates
the child's Q-value so the parent (opposite player) maximises correctly.

Move indices are in canonical (current-player-to-move) coordinates:
from_sq and to_sq are flipped for Black so the network sees a consistent view.
"""
from __future__ import annotations

import math
import chess
import numpy as np
import torch
import torch.nn.functional as F

from .features import board_to_tensor, move_to_index, index_to_move, _flip_sq, NUM_POLICY

_MAX_SIM_DEPTH = 100   # safety cap — prevents runaway simulations on rare positions


class MCTSNode:
    __slots__ = ("prior", "visit_count", "value_sum", "children")

    def __init__(self, prior: float = 1.0) -> None:
        self.prior: float = prior
        self.visit_count: int = 0
        self.value_sum: float = 0.0
        self.children: dict[int, MCTSNode] = {}

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def is_expanded(self) -> bool:
        return bool(self.children)

    def select_child(self, c_puct: float) -> tuple[int, MCTSNode]:
        """PUCT: argmax_a [ -Q(child) + c_puct * P(a) * sqrt(N) / (1 + N(a)) ]."""
        sqrt_n = math.sqrt(max(1, self.visit_count))
        best_score = -float("inf")
        best_action = -1
        best_child: MCTSNode | None = None
        for action, child in self.children.items():
            score = (-child.q_value
                     + c_puct * child.prior * sqrt_n / (1.0 + child.visit_count))
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        return best_action, best_child  # type: ignore[return-value]


# ── helpers ────────────────────────────────────────────────────────────────────

def _net_eval(board: chess.Board, net, device) -> tuple[float, np.ndarray]:
    """One forward pass with legal-move masking.

    Returns (value_from_stm_pov, prob_array[NUM_POLICY]) where only legal
    canonical move indices receive non-zero probability.
    """
    t = board_to_tensor(board)
    with torch.no_grad():
        v, logits = net(t.unsqueeze(0).to(device))

    # Build legal-move mask in canonical coordinates
    legal_idxs = [move_to_index(m, board) for m in board.legal_moves]
    mask = torch.full((NUM_POLICY,), float('-inf'), device=device)
    if legal_idxs:
        mask[legal_idxs] = 0.0

    probs = F.softmax(logits.squeeze(0) + mask, dim=0).cpu().numpy()
    return float(v.squeeze()), probs


def _expand(node: MCTSNode, board: chess.Board, net, device) -> float:
    """Create children for all legal moves. Returns value from board.turn's POV."""
    value, probs = _net_eval(board, net, device)
    for move in board.legal_moves:
        idx = move_to_index(move, board)
        node.children[idx] = MCTSNode(prior=float(probs[idx]))
    return value


def _action_to_move(action: int, board: chess.Board) -> chess.Move:
    """Reconstruct a chess.Move from a canonical action index."""
    from_sq, to_sq = divmod(action, 64)
    # Unflip canonical coordinates back to real board squares for Black
    if board.turn == chess.BLACK:
        from_sq = _flip_sq(from_sq)
        to_sq   = _flip_sq(to_sq)
    piece = board.piece_at(from_sq)
    promotion = None
    if piece and piece.piece_type == chess.PAWN:
        to_rank = chess.square_rank(to_sq)
        if (board.turn == chess.WHITE and to_rank == 7) or \
           (board.turn == chess.BLACK and to_rank == 0):
            promotion = chess.QUEEN
    return chess.Move(from_sq, to_sq, promotion=promotion)


def _terminal_value(board: chess.Board) -> float:
    """Value from board.turn's perspective for a finished game."""
    result = board.result(claim_draw=True)
    if result == "1-0":
        return 1.0 if board.turn == chess.WHITE else -1.0
    if result == "0-1":
        return 1.0 if board.turn == chess.BLACK else -1.0
    return 0.0


def _add_dirichlet_noise(node: MCTSNode,
                          alpha: float = 0.3,
                          eps: float = 0.25) -> None:
    if not node.children:
        return
    noise = np.random.dirichlet([alpha] * len(node.children)).astype(np.float32)
    for i, child in enumerate(node.children.values()):
        child.prior = (1.0 - eps) * child.prior + eps * float(noise[i])


# ── main search ────────────────────────────────────────────────────────────────

def mcts_search(
    board: chess.Board,
    net,
    device,
    n_simulations: int = 100,
    c_puct: float = 1.0,
    dirichlet_alpha: float = 0.3,
    dirichlet_eps: float = 0.25,
    add_noise: bool = True,
) -> np.ndarray:
    """Run PUCT-MCTS from `board`. Returns normalized visit-count vector (NUM_POLICY,)."""
    root = MCTSNode()
    _expand(root, board, net, device)
    if add_noise and root.children:
        _add_dirichlet_noise(root, dirichlet_alpha, dirichlet_eps)

    for _ in range(n_simulations):
        node = root
        sim_board = board.copy(stack=False)
        path: list[MCTSNode] = [root]
        depth = 0

        # ── Selection ─────────────────────────────────────────────────────
        while (node.is_expanded()
               and not sim_board.is_game_over(claim_draw=True)
               and depth < _MAX_SIM_DEPTH):
            action, node = node.select_child(c_puct)
            move = _action_to_move(action, sim_board)
            if move not in sim_board.legal_moves:
                break
            sim_board.push(move)
            path.append(node)
            depth += 1

        # ── Evaluation ────────────────────────────────────────────────────
        if sim_board.is_game_over(claim_draw=True):
            leaf_value = _terminal_value(sim_board)
        else:
            leaf_value = _expand(node, sim_board, net, device)

        # ── Backpropagation ───────────────────────────────────────────────
        # leaf_value is from sim_board.turn's POV = path[-1]'s player's POV.
        # Each step up the tree alternates perspective.
        v = leaf_value
        for n in reversed(path):
            n.visit_count += 1
            n.value_sum += v
            v = -v

    # ── Build policy from visit counts ────────────────────────────────────
    policy = np.zeros(NUM_POLICY, dtype=np.float32)
    for action_idx, child in root.children.items():
        policy[action_idx] = child.visit_count
    total = policy.sum()
    if total > 0:
        policy /= total
    return policy
