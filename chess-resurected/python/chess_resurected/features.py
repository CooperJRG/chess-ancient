"""Convert board positions to neural network input tensors.

Canonical (AlphaZero-style) representation — always from the current player's
perspective so the network can share weights across both colors:

  Planes 0-5:   current player's pieces  (P N B R Q K)
  Planes 6-11:  opponent's pieces        (p n b r q k)
  Plane 12:     side-to-move             (always 1.0 — "it's our turn")
  Planes 13-14: current player's castling rights (kingside, queenside)
  Planes 15-16: opponent's castling rights        (kingside, queenside)
  Plane 17:     en-passant file column

For Black to move the board rank is flipped (sq ^ 56) so that the current
player always "sees" their pieces advancing up the board.
"""

import torch
import chess

NUM_INPUT_PLANES = 18
NUM_POLICY       = 4096   # 64 × 64 (from_sq × 64 + to_sq, both canonical)


def _flip_sq(sq: int) -> int:
    """Flip rank while preserving file (A1↔A8, H1↔H8, …)."""
    return sq ^ 56


def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """Return a (18, 8, 8) float32 tensor in canonical player-to-move view."""
    planes = torch.zeros(NUM_INPUT_PLANES, 8, 8, dtype=torch.float32)
    us   = board.turn
    flip = (us == chess.BLACK)

    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None:
            continue
        csq  = _flip_sq(sq) if flip else sq
        rank = chess.square_rank(csq)
        file = chess.square_file(csq)
        # piece_type: PAWN=1..KING=6 → offset 0-5
        piece_offset = piece.piece_type - 1
        plane = piece_offset if piece.color == us else piece_offset + 6
        planes[plane, rank, file] = 1.0

    # Plane 12: always 1.0 (canonical "our turn")
    planes[12] = 1.0

    # Castling: current player in 13-14, opponent in 15-16
    them = not us
    if board.has_kingside_castling_rights(us):   planes[13] = 1.0
    if board.has_queenside_castling_rights(us):  planes[14] = 1.0
    if board.has_kingside_castling_rights(them):  planes[15] = 1.0
    if board.has_queenside_castling_rights(them): planes[16] = 1.0

    # En-passant: mark the file column (rank is irrelevant — we fill the column)
    if board.ep_square is not None:
        planes[17, :, chess.square_file(board.ep_square)] = 1.0

    return planes


def move_to_index(move: chess.Move, board: chess.Board) -> int:
    """Encode move as from_sq*64+to_sq in canonical (current-player) coordinates."""
    if board.turn == chess.BLACK:
        return _flip_sq(move.from_square) * 64 + _flip_sq(move.to_square)
    return move.from_square * 64 + move.to_square


def index_to_move(idx: int, board: chess.Board) -> chess.Move:
    """Reconstruct a chess.Move from a canonical action index."""
    from_sq = idx // 64
    to_sq   = idx  % 64
    if board.turn == chess.BLACK:
        from_sq = _flip_sq(from_sq)
        to_sq   = _flip_sq(to_sq)
    return chess.Move(from_sq, to_sq)
