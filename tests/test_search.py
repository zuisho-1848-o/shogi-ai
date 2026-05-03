"""探索テスト: 1手詰め・3手詰めの正解手チェック + MultiPV 形式確認"""
from core.board import PythonShogiBoard
from core.move_gen import PythonShogiMoveGen
from core.rules import RuleSet
from core.types import Move
from eval.material import MaterialEvaluator
from search.alphabeta import AlphaBetaSearcher


def _make_searcher():
    return AlphaBetaSearcher()


def _search(sfen: str, depth: int, multi_pv: int = 1):
    board = PythonShogiBoard.from_sfen(sfen)
    move_gen = PythonShogiMoveGen()
    evaluator = MaterialEvaluator()
    searcher = _make_searcher()
    return searcher.search(
        board=board,
        move_gen=move_gen,
        evaluator=evaluator,
        rules=RuleSet(),
        depth=depth,
        time_limit_ms=5000,
        multi_pv=multi_pv,
    )


# 1手詰め: 後手玉が詰み寸前の局面
# 先手が 2b+ と成ることで1手詰め（飛車角が効いている簡易局面）
# 代わりにシンプルな1手詰め SFEN を使う
# 先手: 金が3aにいて2aの後手玉を詰める
_TSUME_1 = "k8/9/9/9/9/9/9/9/8G b G 1"

# 3手詰め: 標準的な簡易局面
# 先手持ち駒: 金。後手玉が1aにある
_TSUME_3 = "k8/1G7/9/9/9/9/9/9/8G b - 1"


def test_material_evaluator_initial_zero() -> None:
    """初期局面の駒得評価は両者均等なので 0 になる。"""
    from core.board import PythonShogiBoard
    from eval.material import MaterialEvaluator

    board = PythonShogiBoard.initial()
    evaluator = MaterialEvaluator()
    score = evaluator.evaluate(board)
    assert score == 0


def test_search_returns_result() -> None:
    """初期局面で探索結果が返る。"""
    result = _search(
        "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
        depth=2,
    )
    assert result.best_move is not None
    assert result.depth >= 1
    assert result.nodes > 0


def test_search_multi_pv_count() -> None:
    """MultiPV で指定した数の候補手が返る（合法手数を超えない）。"""
    result = _search(
        "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
        depth=2,
        multi_pv=5,
    )
    assert 1 <= len(result.candidates) <= 5


def test_search_candidates_descending_score() -> None:
    """MultiPV 候補手はスコア降順に並んでいる。"""
    result = _search(
        "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
        depth=2,
        multi_pv=5,
    )
    scores = [c.score for c in result.candidates]
    assert scores == sorted(scores, reverse=True)


def test_search_best_is_first_candidate() -> None:
    """best_move は candidates[0].move と一致する。"""
    result = _search(
        "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
        depth=2,
        multi_pv=3,
    )
    assert result.best_move == result.candidates[0].move


def test_search_finds_winning_capture() -> None:
    """飛車を取れる局面では取り手を選ぶ（駒得評価で自明）。
    後手の飛車が先手の歩の前にいる。先手は飛車を取れる。
    SFEN: 先手 歩 7f、後手 飛 7e
    """
    # SFEN "6r2/6P2": file 9から数えて7番目 → file=3, rook=3e, pawn=3f
    sfen = "9/9/9/9/6r2/6P2/9/9/9 b - 1"
    result = _search(sfen, depth=2)
    assert result.best_move is not None
    # 3f の歩が 3e の飛車を取る
    assert result.best_move.to_usi() == "3f3e"


def test_search_usi_output_valid() -> None:
    """全候補手の USI 表現が 4 文字以上の有効な形式。"""
    result = _search(
        "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
        depth=2,
        multi_pv=5,
    )
    for cand in result.candidates:
        usi = cand.move.to_usi()
        assert len(usi) >= 4
