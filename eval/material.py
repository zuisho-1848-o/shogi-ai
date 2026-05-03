from __future__ import annotations

import shogi

from core.board import Board, PythonShogiBoard
from core.types import PIECE_VALUES, Color, PieceType
from eval.base import Evaluator

# python-shogi の piece_type 定数は core.PieceType と同じ値なのでそのまま使える
_HAND_PIECE_TYPES = (
    shogi.PAWN,
    shogi.LANCE,
    shogi.KNIGHT,
    shogi.SILVER,
    shogi.GOLD,
    shogi.BISHOP,
    shogi.ROOK,
)


class MaterialEvaluator(Evaluator):
    """盤上の駒得のみによる評価。手番側から見た centipawn スコアを返す。"""

    def evaluate(self, board: Board) -> int:
        assert isinstance(board, PythonShogiBoard)
        b = board.get_shogi_board()
        side = b.turn  # 0=BLACK, 1=WHITE

        score = 0

        # 盤上の駒
        for sq in range(81):
            piece = b.piece_at(sq)
            if piece is None:
                continue
            pt = PieceType(piece.piece_type)
            value = PIECE_VALUES.get(pt, 0)
            if piece.color == side:
                score += value
            else:
                score -= value

        # 持ち駒
        for pt_int in _HAND_PIECE_TYPES:
            pt = PieceType(pt_int)
            value = PIECE_VALUES[pt]
            score += b.pieces_in_hand[side].get(pt_int, 0) * value
            score -= b.pieces_in_hand[1 - side].get(pt_int, 0) * value

        return score
