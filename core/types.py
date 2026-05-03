from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import TypeAlias

# Square: 0-80 の整数。変換式: (9 - file) + (rank - 1) * 9
# 例: 7g (file=7, rank=7) → (9-7) + (7-1)*9 = 56
Square: TypeAlias = int


class Color(IntEnum):
    BLACK = 0  # 先手
    WHITE = 1  # 後手


class PieceType(IntEnum):
    PAWN = 1
    LANCE = 2
    KNIGHT = 3
    SILVER = 4
    GOLD = 5
    BISHOP = 6
    ROOK = 7
    KING = 8
    PRO_PAWN = 9
    PRO_LANCE = 10
    PRO_KNIGHT = 11
    PRO_SILVER = 12
    HORSE = 13   # 馬（成角）= python-shogi PROM_BISHOP
    DRAGON = 14  # 龍（成飛）= python-shogi PROM_ROOK


# 駒の点数（centipawn: 歩=100）。評価関数で使用。
PIECE_VALUES: dict[PieceType, int] = {
    PieceType.PAWN: 100,
    PieceType.LANCE: 430,
    PieceType.KNIGHT: 450,
    PieceType.SILVER: 640,
    PieceType.GOLD: 690,
    PieceType.BISHOP: 890,
    PieceType.ROOK: 1040,
    PieceType.KING: 20000,
    PieceType.PRO_PAWN: 600,
    PieceType.PRO_LANCE: 630,
    PieceType.PRO_KNIGHT: 650,
    PieceType.PRO_SILVER: 680,
    PieceType.HORSE: 1100,
    PieceType.DRAGON: 1300,
}

_USI_CHAR_TO_PIECE: dict[str, PieceType] = {
    "P": PieceType.PAWN,
    "L": PieceType.LANCE,
    "N": PieceType.KNIGHT,
    "S": PieceType.SILVER,
    "G": PieceType.GOLD,
    "B": PieceType.BISHOP,
    "R": PieceType.ROOK,
}
_PIECE_TO_USI_CHAR: dict[PieceType, str] = {v: k for k, v in _USI_CHAR_TO_PIECE.items()}


def sq_from_file_rank(file: int, rank: int) -> Square:
    """file: 1-9（右=1, 左=9）, rank: 1-9（上=1, 下=9）"""
    return (9 - file) + (rank - 1) * 9


def file_of(sq: Square) -> int:
    return 9 - (sq % 9)


def rank_of(sq: Square) -> int:
    return sq // 9 + 1


def _usi_to_sq(s: str) -> Square:
    file = int(s[0])
    rank = ord(s[1]) - ord("a") + 1
    return sq_from_file_rank(file, rank)


def _sq_to_usi(sq: Square) -> str:
    return f"{file_of(sq)}{chr(ord('a') + rank_of(sq) - 1)}"


@dataclass(frozen=True)
class Move:
    from_sq: Square | None       # None は打ち手
    to_sq: Square
    promote: bool = False
    drop_piece_type: PieceType | None = None

    def to_usi(self) -> str:
        if self.drop_piece_type is not None:
            return f"{_PIECE_TO_USI_CHAR[self.drop_piece_type]}*{_sq_to_usi(self.to_sq)}"
        assert self.from_sq is not None
        promo = "+" if self.promote else ""
        return f"{_sq_to_usi(self.from_sq)}{_sq_to_usi(self.to_sq)}{promo}"

    @classmethod
    def from_usi(cls, usi: str) -> Move:
        if "*" in usi:
            return cls(
                from_sq=None,
                to_sq=_usi_to_sq(usi[2:4]),
                drop_piece_type=_USI_CHAR_TO_PIECE[usi[0]],
            )
        return cls(
            from_sq=_usi_to_sq(usi[0:2]),
            to_sq=_usi_to_sq(usi[2:4]),
            promote=len(usi) > 4 and usi[4] == "+",
        )
