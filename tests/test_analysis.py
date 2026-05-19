"""analysis モジュールのテスト"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from analysis.result import format_candidates
from core.types import Move
from eval.material import MaterialEvaluator
from search.base import CandidateMove, SearchResult


# ------------------------------------------------------------------ helpers

def _make_result(n: int) -> SearchResult:
    """テスト用の SearchResult を生成する。"""
    usis = ["7g7f", "8c8d", "2g2f", "8d8e", "2f2e"][:n]
    candidates = [
        CandidateMove(
            move=Move.from_usi(usi),
            score=1000 - i * 200,
            pv=[Move.from_usi(usi)],
        )
        for i, usi in enumerate(usis)
    ]
    return SearchResult(
        best_move=candidates[0].move if candidates else None,
        best_score=candidates[0].score if candidates else 0,
        candidates=candidates,
        depth=3,
        nodes=1000,
    )


# ------------------------------------------------------------------ format_candidates

def test_format_candidates_count() -> None:
    assert len(format_candidates(_make_result(5), max_n=5)) == 5


def test_format_candidates_max_n_truncates() -> None:
    assert len(format_candidates(_make_result(5), max_n=3)) == 3


def test_format_candidates_fields() -> None:
    for i, item in enumerate(format_candidates(_make_result(3))):
        assert item["rank"] == i + 1
        assert isinstance(item["move"], str) and len(item["move"]) >= 4
        assert isinstance(item["score"], int)
        assert isinstance(item["pv"], list)


def test_format_candidates_empty() -> None:
    assert format_candidates(_make_result(0)) == []


def test_format_candidates_scores_descending() -> None:
    items = format_candidates(_make_result(5))
    scores = [c["score"] for c in items]
    assert scores == sorted(scores, reverse=True)


# ------------------------------------------------------------------ eval_graph

def test_eval_graph_creates_png() -> None:
    pytest.importorskip("matplotlib")
    from analysis.eval_graph import save_eval_graph

    scores = [100, 200, -100, 300, -50, 0, 150]
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "eval.png"
        save_eval_graph(scores, out, title="テスト")
        assert out.exists()
        assert out.stat().st_size > 0


def test_eval_graph_empty_scores() -> None:
    pytest.importorskip("matplotlib")
    from analysis.eval_graph import save_eval_graph

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "empty.png"
        save_eval_graph([], out)
        assert out.exists()


def test_eval_graph_creates_parent_dirs() -> None:
    pytest.importorskip("matplotlib")
    from analysis.eval_graph import save_eval_graph

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "subdir" / "nested" / "eval.png"
        save_eval_graph([0, 100, -100], out)
        assert out.exists()


# ------------------------------------------------------------------ kifu_analyzer

def test_kifu_analyzer_length() -> None:
    from analysis.kifu_analyzer import analyze_game

    moves = ["7g7f", "3c3d"]
    results = analyze_game(moves, MaterialEvaluator(), depth=0)
    assert len(results) == len(moves)


def test_kifu_analyzer_ply_numbers() -> None:
    from analysis.kifu_analyzer import analyze_game

    moves = ["7g7f", "3c3d", "2g2f"]
    results = analyze_game(moves, MaterialEvaluator(), depth=0)
    for i, r in enumerate(results, start=1):
        assert r.ply == i


def test_kifu_analyzer_move_field() -> None:
    from analysis.kifu_analyzer import analyze_game

    moves = ["7g7f", "3c3d"]
    results = analyze_game(moves, MaterialEvaluator(), depth=0)
    for r, m in zip(results, moves):
        assert r.move == m


def test_kifu_analyzer_eval_types() -> None:
    from analysis.kifu_analyzer import analyze_game

    results = analyze_game(["7g7f"], MaterialEvaluator(), depth=0)
    assert isinstance(results[0].eval_before, int)
    assert isinstance(results[0].eval_after, int)


def test_kifu_analyzer_empty_game() -> None:
    from analysis.kifu_analyzer import analyze_game

    assert analyze_game([], MaterialEvaluator()) == []


def test_kifu_analyzer_no_candidates_when_depth_zero() -> None:
    from analysis.kifu_analyzer import analyze_game

    results = analyze_game(["7g7f"], MaterialEvaluator(), depth=0)
    assert results[0].candidates == []
