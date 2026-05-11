"""KP評価関数 (King-Piece テーブル、Bonanza方式)。

テーブル形状: float32[81][PIECE_FEAT_SIZE]
  先手玉マス × 駒特徴 → スコア寄与 (centipawn)

評価値 = Σ table[k_b][feat] (先手玉視点の全駒)
       - Σ table[k_w_mirror][feat] (後手玉視点の全駒)
この符号は「先手から見たスコア」。Negamax形式で返す。

重みファイル: models/kpp.npz (なければPSTにフォールバック)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import shogi

from core.board import Board, PythonShogiBoard
from eval.base import Evaluator
from eval.pst import PSTEvaluator
from train.dataset import PIECE_FEAT_SIZE, compute_kp_indices

_DEFAULT_MODEL = Path("models/kpp.npz")


class KPPEvaluator(Evaluator):
    """KPテーブルによる評価。手番側から見たcentipawnスコアを返す。"""

    def __init__(
        self,
        table: np.ndarray,
        fallback: Evaluator | None = None,
    ) -> None:
        # table: float32[81][PIECE_FEAT_SIZE]
        assert table.shape == (81, PIECE_FEAT_SIZE), table.shape
        self._table = table.astype(np.float32)
        self._fallback = fallback or PSTEvaluator()

    # ------------------------------------------------------------------ #
    # 評価                                                                  #
    # ------------------------------------------------------------------ #

    def evaluate(self, board: Board) -> int:
        assert isinstance(board, PythonShogiBoard)
        b = board.get_shogi_board()
        side = b.turn  # 0=BLACK, 1=WHITE

        score_b = self._kp_score(b, shogi.BLACK)
        score_w = self._kp_score(b, shogi.WHITE)
        score_black = score_b - score_w  # 先手視点スコア

        return int(score_black) if side == shogi.BLACK else -int(score_black)

    def _kp_score(self, b: shogi.Board, king_color: int) -> float:
        king_sq, indices = compute_kp_indices(b, king_color)
        if not indices:
            return 0.0
        return float(self._table[king_sq, indices].sum())

    # ------------------------------------------------------------------ #
    # 保存 / 読み込み                                                       #
    # ------------------------------------------------------------------ #

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, table=self._table)

    @classmethod
    def load(cls, path: Path) -> KPPEvaluator:
        d = np.load(path)
        return cls(table=d["table"].astype(np.float32))

    @classmethod
    def load_or_fallback(cls, path: Path = _DEFAULT_MODEL) -> KPPEvaluator | PSTEvaluator:
        """モデルがあればKPP、なければPSTを返す。"""
        if path.exists():
            return cls.load(path)
        return PSTEvaluator()

    # ------------------------------------------------------------------ #
    # ファクトリ                                                            #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_zeros(cls) -> KPPEvaluator:
        """ゼロ初期化テーブル（学習前ベースライン）。"""
        return cls(np.zeros((81, PIECE_FEAT_SIZE), dtype=np.float32))

    @classmethod
    def from_pst_values(cls) -> KPPEvaluator:
        """
        PSTの駒得+位置ボーナスをKPテーブルの初期値に変換する。

        KP評価は score = score_from_black_king - score_from_white_king で計算される。
        白玉視点では盤面が鏡像化されるため、自分の駒の貢献がそのまま反映される。
        そのため "自分の駒" (own_feat) にのみ正の値を設定し、
        "相手の駒" (opp_feat) は0にしてよい (白玉視点で自動的に加算される)。

        手駒は位置ボーナスなし、基本価値のみ。
        """
        import shogi as _shogi

        from core.types import PIECE_VALUES, PieceType
        from core.types import rank_of
        from eval.pst import _RANK_BONUS
        from train.dataset import (
            BOARD_FEAT, HAND_FEAT_PER_COLOR,
            _HAND_OFFSETS, _HAND_TYPES, _HAND_MAX,
        )

        table = np.zeros((81, PIECE_FEAT_SIZE), dtype=np.float32)

        _PT_MAP = {
            _shogi.PAWN: PieceType.PAWN, _shogi.LANCE: PieceType.LANCE,
            _shogi.KNIGHT: PieceType.KNIGHT, _shogi.SILVER: PieceType.SILVER,
            _shogi.GOLD: PieceType.GOLD, _shogi.BISHOP: PieceType.BISHOP,
            _shogi.ROOK: PieceType.ROOK,
            _shogi.PROM_PAWN: PieceType.PRO_PAWN, _shogi.PROM_LANCE: PieceType.PRO_LANCE,
            _shogi.PROM_KNIGHT: PieceType.PRO_KNIGHT, _shogi.PROM_SILVER: PieceType.PRO_SILVER,
            _shogi.PROM_BISHOP: PieceType.HORSE, _shogi.PROM_ROOK: PieceType.DRAGON,
        }
        _KP_TYPES = {
            _shogi.PAWN: 0, _shogi.LANCE: 1, _shogi.KNIGHT: 2,
            _shogi.SILVER: 3, _shogi.GOLD: 4, _shogi.BISHOP: 5, _shogi.ROOK: 6,
            _shogi.PROM_PAWN: 7, _shogi.PROM_LANCE: 8, _shogi.PROM_KNIGHT: 9,
            _shogi.PROM_SILVER: 10, _shogi.PROM_BISHOP: 11, _shogi.PROM_ROOK: 12,
        }

        # 盤上の駒: "自分の駒" 特徴に PST 値を設定 (全玉マス共通)
        for pt_int, kp_idx in _KP_TYPES.items():
            core_pt = _PT_MAP.get(pt_int)
            if core_pt is None:
                continue
            base_value = PIECE_VALUES.get(core_pt, 0)
            rank_bonus_table = _RANK_BONUS.get(core_pt, (0,) * 9)

            for piece_sq in range(81):
                rank_idx = rank_of(piece_sq) - 1
                rank_idx = max(0, min(8, rank_idx))
                pst_value = base_value + rank_bonus_table[rank_idx]
                # own piece feature (color=0)
                feat_own = kp_idx * 81 + piece_sq
                table[:, feat_own] = pst_value  # 全玉マスに同じ値

        # 手駒: "自分の手駒" 特徴に基本価値を設定
        for pt_int in _HAND_TYPES:
            core_pt = _PT_MAP.get(pt_int)
            if core_pt is None:
                continue
            base_value = PIECE_VALUES.get(core_pt, 0)
            own_offset = BOARD_FEAT  # hand_idx=0 = 自分の手駒
            hand_base = own_offset + _HAND_OFFSETS[pt_int]
            max_count = _HAND_MAX[pt_int]
            for c in range(max_count):
                table[:, hand_base + c] = float(base_value)

        return cls(table)
