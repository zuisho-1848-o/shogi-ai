from __future__ import annotations

import shogi

from core.board import Board, PythonShogiBoard
from core.types import PIECE_VALUES, Color, PieceType, rank_of
from eval.base import Evaluator

# ランク別ボーナス（インデックス 0 = rank1 = 先手の成り込みゾーン）
# すべて先手視点で定義。後手は rank_idx = 9 - rank_of(sq) に変換して参照する。
_RANK_BONUS: dict[PieceType, tuple[int, ...]] = {
    PieceType.PAWN:        (50, 40, 30, 20, 10,  5,  0, -10, -20),
    PieceType.LANCE:       (60, 50, 40, 30, 20, 10,  0,  -5, -15),
    PieceType.KNIGHT:      (0,  50, 40, 30, 20, 10,  0,  -5, -15),
    PieceType.SILVER:      (15, 15, 15, 10,  5,  0, -5, -10, -15),
    PieceType.GOLD:        (10, 10, 10,  5,  0,  0, -5, -10, -15),
    PieceType.BISHOP:      ( 0,  5,  5,  5,  5,  5,  5,   5,   0),
    PieceType.ROOK:        ( 5, 10,  5,  0,  0,  5, 10,  10,   5),
    PieceType.KING:        (-50,-40,-30,-20,-10,  0,  5,  15,  20),
    PieceType.PRO_PAWN:    (10, 10, 10,  8,  5,  0, -5, -10, -15),
    PieceType.PRO_LANCE:   (10, 10, 10,  8,  5,  0, -5, -10, -15),
    PieceType.PRO_KNIGHT:  (10, 10, 10,  8,  5,  0, -5, -10, -15),
    PieceType.PRO_SILVER:  (10, 10, 10,  8,  5,  0, -5, -10, -15),
    PieceType.HORSE:       ( 5, 10, 10, 10,  5,  5,  5,   5,   5),
    PieceType.DRAGON:      (10, 15, 10,  5,  5,  5, 10,  10,   5),
}

_HAND_PIECE_TYPES = (
    shogi.PAWN,
    shogi.LANCE,
    shogi.KNIGHT,
    shogi.SILVER,
    shogi.GOLD,
    shogi.BISHOP,
    shogi.ROOK,
)


class PSTEvaluator(Evaluator):
    """駒得 + 位置ボーナステーブル（PST）による評価。手番側スコアを返す。"""

    def evaluate(self, board: Board) -> int:
        assert isinstance(board, PythonShogiBoard)
        b = board.get_shogi_board()
        side = b.turn  # 0=BLACK, 1=WHITE

        score = 0

        for sq in range(81):
            piece = b.piece_at(sq)
            if piece is None:
                continue
            try:
                pt = PieceType(piece.piece_type)
            except ValueError:
                continue

            base_value = PIECE_VALUES.get(pt, 0)
            rank_idx = (rank_of(sq) - 1) if piece.color == 0 else (9 - rank_of(sq))
            rank_idx = max(0, min(8, rank_idx))
            pst_bonus = _RANK_BONUS.get(pt, (0,) * 9)[rank_idx]

            total = base_value + pst_bonus
            if piece.color == side:
                score += total
            else:
                score -= total

        # 持ち駒（PST ボーナスなし、駒得のみ）
        for pt_int in _HAND_PIECE_TYPES:
            pt = PieceType(pt_int)
            value = PIECE_VALUES[pt]
            score += b.pieces_in_hand[side].get(pt_int, 0) * value
            score -= b.pieces_in_hand[1 - side].get(pt_int, 0) * value

        return score
