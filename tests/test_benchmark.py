"""ベンチマークインフラのテスト。"""
from __future__ import annotations

from benchmark.tsume import DEFAULT_PROBLEMS, TsumeBenchmark, TsumeProblem
from benchmark.self_play import SelfPlay, SelfPlayConfig, SelfPlayResult
from eval.pst import PSTEvaluator


# ------------------------------------------------------------------ TsumeBenchmark

def test_tsume_benchmark_runs() -> None:
    """ベンチマークが例外なく終了する。"""
    bench = TsumeBenchmark(time_limit_ms=5_000)
    results = bench.run()
    assert isinstance(results, list)
    assert len(results) == len(DEFAULT_PROBLEMS)


def test_tsume_result_has_fields() -> None:
    bench = TsumeBenchmark(time_limit_ms=5_000)
    results = bench.run()
    for r in results:
        assert hasattr(r, "solved")
        assert hasattr(r, "found_move")
        assert hasattr(r, "elapsed_ms")
        assert hasattr(r, "nodes")
        assert r.elapsed_ms >= 0
        assert r.nodes >= 0


def test_tsume_1te_solved() -> None:
    """検証済みの1手詰め問題を正しく解ける。"""
    problem = TsumeProblem(
        sfen="6S1k/9/9/9/9/9/9/9/9 b G 1",
        mate_in=1,
        description="test: 1手詰め",
        expected_move="G*2b",
    )
    bench = TsumeBenchmark(problems=[problem], time_limit_ms=10_000)
    results = bench.run()
    assert results[0].solved
    assert results[0].found_move == "G*2b"


def test_tsume_report_returns_string() -> None:
    bench = TsumeBenchmark(time_limit_ms=3_000)
    results = bench.run()
    report = TsumeBenchmark.report(results)
    assert isinstance(report, str)
    assert "手詰め" in report


# ------------------------------------------------------------------ SelfPlay

def test_self_play_runs_small() -> None:
    """2局だけ対局して結果が返ることを確認する。"""
    cfg = SelfPlayConfig(depth=2, time_limit_ms=500, max_moves=50)
    sp = SelfPlay(PSTEvaluator(), PSTEvaluator(), cfg)
    result = sp.run(n_games=2)

    assert isinstance(result, SelfPlayResult)
    assert result.n_games == 2
    assert result.engine1_wins + result.engine2_wins + result.draws == 2


def test_self_play_win_rate_range() -> None:
    cfg = SelfPlayConfig(depth=2, time_limit_ms=500, max_moves=50)
    sp = SelfPlay(PSTEvaluator(), PSTEvaluator(), cfg)
    result = sp.run(n_games=4)

    assert 0.0 <= result.engine1_win_rate <= 1.0


def test_self_play_str() -> None:
    cfg = SelfPlayConfig(depth=2, time_limit_ms=500, max_moves=30)
    sp = SelfPlay(PSTEvaluator(), PSTEvaluator(), cfg)
    result = sp.run(n_games=2)
    s = str(result)
    assert "Engine1" in s
    assert "Engine2" in s
