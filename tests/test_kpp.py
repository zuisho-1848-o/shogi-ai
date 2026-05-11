"""KPP評価関数のテスト。"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import shogi

from core.board import PythonShogiBoard
from eval.kpp import KPPEvaluator
from train.dataset import (
    PIECE_FEAT_SIZE,
    compute_kp_indices,
    load_csa_file,
)


# ------------------------------------------------------------------ #
# 特徴量テスト                                                          #
# ------------------------------------------------------------------ #

class TestKPFeatures:
    def test_piece_feat_size(self) -> None:
        # 2色×13種×81マス + 2色×38持ち駒 = 2106 + 76 = 2182
        assert PIECE_FEAT_SIZE == 2182

    def test_startpos_black_king_indices(self) -> None:
        b = shogi.Board()
        king_sq, indices = compute_kp_indices(b, shogi.BLACK)
        # 先手玉は初期局面で sq=76 (5i)
        assert king_sq == 76
        # 初期局面: 先手19駒 + 後手19駒 = 38枚の非王駒
        assert len(indices) == 38
        # 全インデックスが有効範囲内
        assert all(0 <= i < PIECE_FEAT_SIZE for i in indices)

    def test_startpos_white_king_indices(self) -> None:
        b = shogi.Board()
        king_sq, indices = compute_kp_indices(b, shogi.WHITE)
        # 後手玉は鏡像化される (sq=4 → mirror=76)
        assert king_sq == 76
        assert len(indices) == 38

    def test_indices_no_duplicates(self) -> None:
        b = shogi.Board()
        _, idx_b = compute_kp_indices(b, shogi.BLACK)
        _, idx_w = compute_kp_indices(b, shogi.WHITE)
        # 初期局面では持ち駒なし → 重複なし
        assert len(set(idx_b)) == len(idx_b)
        assert len(set(idx_w)) == len(idx_w)

    def test_custom_position_with_hand(self) -> None:
        # 初期局面から7g7fで歩を動かした局面 (持ち駒なし、盤上の駒数は変わらず38)
        b = shogi.Board()
        b.push_usi("7g7f")
        _, idx_b = compute_kp_indices(b, shogi.BLACK)
        assert len(idx_b) == 38


# ------------------------------------------------------------------ #
# KPPEvaluator テスト                                                  #
# ------------------------------------------------------------------ #

class TestKPPEvaluator:
    def test_from_zeros_evaluate(self) -> None:
        ev = KPPEvaluator.from_zeros()
        board = PythonShogiBoard.from_sfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1")
        score = ev.evaluate(board)
        # ゼロ初期化なのでスコアは0
        assert score == 0

    def test_from_pst_values_evaluate(self) -> None:
        ev = KPPEvaluator.from_pst_values()
        board = PythonShogiBoard.from_sfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1")
        # 対称局面 → スコアはほぼ0（先手後手が対称なので）
        score = ev.evaluate(board)
        # ±50以内に収まるはず（完全に0にはならない場合がある）
        assert abs(score) < 100

    def test_material_advantage_reflected(self) -> None:
        ev = KPPEvaluator.from_pst_values()
        # 後手の飛車を先手が持ち駒にした局面 (後手は飛車なし、先手は持ち駒に飛車)
        # 7b の飛車がない + 先手持ち駒に r
        board_adv = PythonShogiBoard.from_sfen("lnsgkgsnl/7b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b r 1")
        # 対称な初期局面
        board_start = PythonShogiBoard.from_sfen("lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1")
        score_adv = ev.evaluate(board_adv)
        score_start = ev.evaluate(board_start)
        # 飛車を1枚得している → 圧倒的優勢
        assert score_adv > score_start

    def test_turn_perspective(self) -> None:
        """手番側から見たスコアが正しく符号反転されるか。"""
        ev = KPPEvaluator.from_pst_values()
        # 先手が後手の飛車を持ち駒にした局面 (先手有利)
        board_b = PythonShogiBoard.from_sfen("lnsgkgsnl/7b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b r 1")
        board_w = PythonShogiBoard.from_sfen("lnsgkgsnl/7b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w r 1")
        score_b = ev.evaluate(board_b)  # 先手番 → 先手有利 → 正
        score_w = ev.evaluate(board_w)  # 後手番 → 先手有利だが後手視点 → 負
        assert score_b > 0
        assert score_w < 0

    def test_save_and_load(self) -> None:
        ev = KPPEvaluator.from_pst_values()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "kpp.npz"
            ev.save(path)
            ev2 = KPPEvaluator.load(path)
        assert ev2._table.shape == (81, PIECE_FEAT_SIZE)
        np.testing.assert_allclose(ev._table, ev2._table)

    def test_load_or_fallback_missing(self) -> None:
        """モデルが存在しない場合はPSTにフォールバックする。"""
        from eval.pst import PSTEvaluator
        result = KPPEvaluator.load_or_fallback(Path("models/nonexistent.npz"))
        assert isinstance(result, PSTEvaluator)

    def test_load_or_fallback_exists(self) -> None:
        """models/kpp.npz が存在すればKPPEvaluatorを返す。"""
        path = Path("models/kpp.npz")
        if not path.exists():
            pytest.skip("models/kpp.npz が存在しないためスキップ")
        result = KPPEvaluator.load_or_fallback(path)
        assert isinstance(result, KPPEvaluator)

    def test_evaluate_does_not_crash_various_positions(self) -> None:
        """各種局面でクラッシュしないか確認。"""
        ev = KPPEvaluator.from_pst_values()
        sfens = [
            "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
            "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1",
            "8k/9/9/9/9/9/9/9/K8 b - 1",   # 玉だけ
            "8k/9/9/9/9/9/9/9/K8 b R 1",    # 持ち駒あり (4パーツ正しいSFEN)
        ]
        for sfen in sfens:
            board = PythonShogiBoard.from_sfen(sfen)
            score = ev.evaluate(board)
            assert isinstance(score, int)


# ------------------------------------------------------------------ #
# CSAパーサーのスモークテスト                                            #
# ------------------------------------------------------------------ #

class TestCSAParser:
    def test_load_nonexistent_file(self) -> None:
        result = load_csa_file(Path("/nonexistent/file.csa"))
        assert result == []

    def test_load_minimal_csa(self, tmp_path: Path) -> None:
        # 最小限のCSAデータで1局面だけ読めることを確認
        csa_content = """\
V2.2
N+テスト先手
N-テスト後手
$START_TIME:2020/01/01 00:00:00
P1-KY-KE-GI-KI-OU-KI-GI-KE-KY
P2 * -HI *  *  *  *  * -KA *
P3-FU-FU-FU-FU-FU-FU-FU-FU-FU
P4 *  *  *  *  *  *  *  *  *
P5 *  *  *  *  *  *  *  *  *
P6 *  *  *  *  *  *  *  *  *
P7+FU+FU+FU+FU+FU+FU+FU+FU+FU
P8 * +KA *  *  *  *  * +HI *
P9+KY+KE+GI+KI+OU+KI+GI+KE+KY
+
+7776FU
-3334FU
%TORYO
"""
        csa_file = tmp_path / "test.csa"
        csa_file.write_text(csa_content, encoding="utf-8")
        result = load_csa_file(csa_file)
        # 2手指されたので2局面が取れるはず（先手が投了した=後手勝）
        assert len(result) >= 1
        # 先手が投了 → 後手が最後に指した後に先手投了 → 後手勝 = -1.0
        assert result[0][1] == -1.0
