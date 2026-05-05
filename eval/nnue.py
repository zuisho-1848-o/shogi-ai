"""NNUE 評価関数 (HalfKP-lite)。

特徴量: 盤面 (color×14×81=2268次元) + 持ち駒 (2×7=14次元) = 2282次元
ネットワーク: Linear(2282,256) → ReLU → Linear(256,32) → ReLU → Linear(32,1) × 600
重みファイル: models/nnue.npz (なければ PST にフォールバック)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import shogi

from core.board import Board, PythonShogiBoard
from core.types import Color
from eval.base import Evaluator
from eval.pst import PSTEvaluator

_NUM_PT = 14
_NUM_SQ = 81
_BOARD_FEAT = 2 * _NUM_PT * _NUM_SQ   # 2268
_HAND_PT = (
    shogi.PAWN, shogi.LANCE, shogi.KNIGHT, shogi.SILVER,
    shogi.GOLD, shogi.BISHOP, shogi.ROOK,
)
_HAND_MAX = {
    shogi.PAWN: 18, shogi.LANCE: 4, shogi.KNIGHT: 4,
    shogi.SILVER: 4, shogi.GOLD: 4, shogi.BISHOP: 2, shogi.ROOK: 2,
}
_HAND_FEAT = 2 * len(_HAND_PT)        # 14
INPUT_SIZE = _BOARD_FEAT + _HAND_FEAT  # 2282
L1_SIZE = 256
L2_SIZE = 32
OUTPUT_SCALE = 600.0

_DEFAULT_MODEL = Path("models/nnue.npz")


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


class NNUENetwork:
    """3層 MLP (numpy)。weights は .npz から読み込む。"""

    def __init__(
        self,
        w1: np.ndarray, b1: np.ndarray,
        w2: np.ndarray, b2: np.ndarray,
        w3: np.ndarray, b3: np.ndarray,
    ) -> None:
        self.w1 = w1  # (INPUT_SIZE, L1_SIZE)
        self.b1 = b1  # (L1_SIZE,)
        self.w2 = w2  # (L1_SIZE, L2_SIZE)
        self.b2 = b2  # (L2_SIZE,)
        self.w3 = w3  # (L2_SIZE, 1)
        self.b3 = b3  # (1,)

    def forward(self, x: np.ndarray) -> float:
        h1 = _relu(x @ self.w1 + self.b1)
        h2 = _relu(h1 @ self.w2 + self.b2)
        return float((h2 @ self.w3 + self.b3)[0]) * OUTPUT_SCALE

    @classmethod
    def load(cls, path: Path) -> NNUENetwork:
        d = np.load(path)
        to_f32 = lambda k: d[k].astype(np.float32)  # noqa: E731
        return cls(to_f32("w1"), to_f32("b1"), to_f32("w2"),
                   to_f32("b2"), to_f32("w3"), to_f32("b3"))


def extract_features(board: PythonShogiBoard) -> np.ndarray:
    """盤面 → INPUT_SIZE 次元の特徴ベクトル (float32)。"""
    b = board.get_shogi_board()
    feat = np.zeros(INPUT_SIZE, dtype=np.float32)

    for sq in range(_NUM_SQ):
        piece = b.piece_at(sq)
        if piece is None:
            continue
        pt = piece.piece_type
        if not 1 <= pt <= 14:
            continue
        feat[piece.color * _NUM_PT * _NUM_SQ + (pt - 1) * _NUM_SQ + sq] = 1.0

    for ci in range(2):
        for ti, pt_int in enumerate(_HAND_PT):
            cnt = b.pieces_in_hand[ci].get(pt_int, 0)
            feat[_BOARD_FEAT + ci * len(_HAND_PT) + ti] = cnt / _HAND_MAX[pt_int]

    return feat


class NNUEEvaluator(Evaluator):
    """NNUE 評価。models/nnue.npz が存在すれば NN 推論、なければ PST にフォールバック。"""

    def __init__(self, model_path: Path = _DEFAULT_MODEL) -> None:
        self._net: NNUENetwork | None = None
        self._fallback = PSTEvaluator()
        if model_path.exists():
            try:
                self._net = NNUENetwork.load(model_path)
            except Exception:
                pass

    @property
    def has_model(self) -> bool:
        return self._net is not None

    def evaluate(self, board: Board) -> int:
        assert isinstance(board, PythonShogiBoard)
        if self._net is None:
            return self._fallback.evaluate(board)

        score = self._net.forward(extract_features(board))
        # ネットワーク出力は BLACK 視点。手番側視点に変換。
        if board.turn == Color.WHITE:
            score = -score
        return int(score)
