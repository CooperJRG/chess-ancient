"""Adapters from PositionRecord rows to neural-network tensors."""

from __future__ import annotations

import chess
import torch
from torch.utils.data import Dataset

from .features import NUM_POLICY, board_to_tensor, move_to_index
from .records import PositionRecord


def legal_mask_from_uci(board: chess.Board, legal_moves_uci: tuple[str, ...]) -> torch.Tensor:
    mask = torch.zeros(NUM_POLICY, dtype=torch.bool)
    source = legal_moves_uci or tuple(m.uci() for m in board.legal_moves)
    for uci in source:
        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            continue
        mask[move_to_index(move, board)] = True
    return mask


def policy_index(record: PositionRecord, board: chess.Board) -> int:
    if record.move_uci is None:
        raise ValueError("record has no move_uci policy target")
    return move_to_index(chess.Move.from_uci(record.move_uci), board)


class PositionRecordDataset(Dataset):
    def __init__(self, records: list[PositionRecord]):
        self.records = [r for r in records if r.move_uci is not None and r.result is not None]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        board = chess.Board(record.fen)
        return (
            board_to_tensor(board),
            torch.tensor(policy_index(record, board), dtype=torch.long),
            legal_mask_from_uci(board, record.legal_moves_uci),
            torch.tensor([float(record.result)], dtype=torch.float32),
        )
