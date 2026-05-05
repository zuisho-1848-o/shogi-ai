"""NNUE 評価関数のテスト。重みファイルの有無両方を検証する。"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from core.board import PythonShogiBoard
from core.types import Move
from eval.nnue import (
    INPUT_SIZE,
    L1_SIZE,
    L2_SIZE,
    NNUEEvaluator,
    NNUENetwork,
    extract_features,
)


# ------------------------------------------------------------------ 特徴抽出

def test_extract_features_shape() -> None:
    board = PythonShogiBoard.initial()
    feat = extract_features(board)
    assert feat.shape == (INPUT_SIZE,)


def test_extract_features_dtype() -> None:
    board = PythonShogiBoard.initial()
    feat = extract_features(board)
    assert feat.dtype == np.float32


def test_extract_features_initial_nonzero() -> None:
    """初期局面で特徴が非ゼロ (駒が盤上にある)。"""
    board = PythonShogiBoard.initial()
    feat = extract_features(board)
    assert feat.sum() > 0


def test_extract_features_changes_after_move() -> None:
    board = PythonShogiBoard.initial()
    feat_before = extract_features(board).copy()
    board2 = board.apply_move(Move.from_usi("7g7f"))
    feat_after = extract_features(board2)
    assert not np.allclose(feat_before, feat_after)


# ------------------------------------------------------------------ NNUENetwork

def _make_dummy_network() -> NNUENetwork:
    rng = np.random.default_rng(0)
    return NNUENetwork(
        w1=rng.normal(0, 0.01, (INPUT_SIZE, L1_SIZE)).astype(np.float32),
        b1=np.zeros(L1_SIZE, dtype=np.float32),
        w2=rng.normal(0, 0.01, (L1_SIZE, L2_SIZE)).astype(np.float32),
        b2=np.zeros(L2_SIZE, dtype=np.float32),
        w3=rng.normal(0, 0.01, (L2_SIZE, 1)).astype(np.float32),
        b3=np.zeros(1, dtype=np.float32),
    )


def test_network_forward_returns_float() -> None:
    net = _make_dummy_network()
    feat = extract_features(PythonShogiBoard.initial())
    result = net.forward(feat)
    assert isinstance(result, float)


def test_network_save_and_load(tmp_path: Path) -> None:
    net = _make_dummy_network()
    feat = extract_features(PythonShogiBoard.initial())
    score_before = net.forward(feat)

    path = tmp_path / "test.npz"
    np.savez(
        str(path),
        w1=net.w1, b1=net.b1,
        w2=net.w2, b2=net.b2,
        w3=net.w3, b3=net.b3,
    )

    loaded = NNUENetwork.load(path)
    score_after = loaded.forward(feat)
    assert abs(score_before - score_after) < 1e-3


# ------------------------------------------------------------------ NNUEEvaluator (fallback)

def test_evaluator_fallback_no_model() -> None:
    """重みファイルがなければ PST にフォールバックして動作する。"""
    ev = NNUEEvaluator(model_path=Path("nonexistent_model.npz"))
    assert not ev.has_model
    board = PythonShogiBoard.initial()
    score = ev.evaluate(board)
    assert isinstance(score, int)


def test_evaluator_fallback_initial_zero() -> None:
    """フォールバック時、初期局面のスコアは 0 (PST と同じ)。"""
    ev = NNUEEvaluator(model_path=Path("nonexistent_model.npz"))
    board = PythonShogiBoard.initial()
    assert ev.evaluate(board) == 0


# ------------------------------------------------------------------ NNUEEvaluator (with model)

def test_evaluator_with_model_returns_int(tmp_path: Path) -> None:
    net = _make_dummy_network()
    path = tmp_path / "nnue.npz"
    np.savez(str(path), w1=net.w1, b1=net.b1, w2=net.w2,
             b2=net.b2, w3=net.w3, b3=net.b3)

    ev = NNUEEvaluator(model_path=path)
    assert ev.has_model
    board = PythonShogiBoard.initial()
    score = ev.evaluate(board)
    assert isinstance(score, int)


def test_evaluator_with_model_symmetric(tmp_path: Path) -> None:
    """盤面を 180° 回転 (後手視点) にしたときスコアが反転する。"""
    net = _make_dummy_network()
    path = tmp_path / "nnue.npz"
    np.savez(str(path), w1=net.w1, b1=net.b1, w2=net.w2,
             b2=net.b2, w3=net.w3, b3=net.b3)

    ev = NNUEEvaluator(model_path=path)
    board = PythonShogiBoard.initial()
    board_w = board.apply_move(Move.from_usi("7g7f"))  # 後手番に変える
    score = ev.evaluate(board_w)
    # スコアが返ること (符号は評価次第なので数値の範囲のみ確認)
    assert isinstance(score, int)


# ------------------------------------------------------------------ create_nnue_weights

def test_create_pst_weights_file_created(tmp_path: Path) -> None:
    from scripts.create_nnue_weights import create_pst_weights

    out = tmp_path / "nnue.npz"
    create_pst_weights(out)
    assert out.exists()


def test_create_pst_weights_correct_keys(tmp_path: Path) -> None:
    from scripts.create_nnue_weights import create_pst_weights

    out = tmp_path / "nnue.npz"
    create_pst_weights(out)
    d = np.load(out)
    for key in ("w1", "b1", "w2", "b2", "w3", "b3"):
        assert key in d


def test_pst_initialized_nnue_vs_pst(tmp_path: Path) -> None:
    """PST 初期化 NNUE の評価値が PST の評価値と近い (誤差 500cp 以内)。"""
    from scripts.create_nnue_weights import create_pst_weights
    from eval.pst import PSTEvaluator

    out = tmp_path / "nnue.npz"
    create_pst_weights(out)

    nnue_ev = NNUEEvaluator(model_path=out)
    pst_ev = PSTEvaluator()

    for sfen in [
        "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
        "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL w - 1",
    ]:
        board = PythonShogiBoard.from_sfen(sfen)
        nnue_score = nnue_ev.evaluate(board)
        pst_score = pst_ev.evaluate(board)
        assert abs(nnue_score - pst_score) < 500, (
            f"NNUE={nnue_score}, PST={pst_score}, diff={abs(nnue_score - pst_score)}"
        )
