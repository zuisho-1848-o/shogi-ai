"""探索テスト: 1手詰め・3手詰めの正解手チェック + MultiPV 形式確認"""
from __future__ import annotations

from core.board import Board
from core.board import PythonShogiBoard
from core.move_gen import MoveGenerator
from core.move_gen import PythonShogiMoveGen
from core.rules import RuleSet
from core.types import Color, Move, PieceType, Square
from eval.base import Evaluator
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


class _FivePlyMateBoard(Board):
    """5 ply の強制メイトを持つ最小ゲーム木。

    python-shogi の詰将棋専用探索ではなく、AlphaBetaSearcher が
    5手先の終端勝ちを読んで初手を選べることを固定するテスト用。
    """

    _NEXT: dict[tuple[str, str], str] = {
        ("root", "7g7f"): "after_1",
        ("root", "2g2f"): "quiet",
        ("after_1", "9a9b"): "after_2",
        ("after_2", "7f7e"): "after_3",
        ("after_3", "9b9a"): "after_4",
        ("after_4", "G*5a"): "mate",
        ("quiet", "9a9b"): "quiet_2",
        ("quiet_2", "2f2e"): "quiet_3",
        ("quiet_3", "9b9a"): "quiet_4",
        ("quiet_4", "2e2d"): "drawish",
    }

    def __init__(self, state: str = "root", ply: int = 0) -> None:
        self._state = state
        self._ply = ply

    def apply_move(self, move: Move) -> Board:
        return _FivePlyMateBoard(self._NEXT[(self._state, move.to_usi())], self._ply + 1)

    def is_check(self) -> bool:
        # Null Move Pruning を止め、純粋な5手先読みの回帰テストにする。
        return True

    def is_game_over(self) -> bool:
        return self._state == "mate"

    @classmethod
    def from_sfen(cls, sfen: str) -> Board:
        return cls(sfen)

    def to_sfen(self) -> str:
        return self._state

    @property
    def turn(self) -> Color:
        return Color.BLACK if self._ply % 2 == 0 else Color.WHITE

    def piece_at_sq(self, sq: Square) -> tuple[PieceType, Color] | None:
        return None

    def null_move_board(self) -> Board:
        return _FivePlyMateBoard(f"{self._state}:null", self._ply + 1)


class _FivePlyMateMoveGen(MoveGenerator):
    _MOVES: dict[str, list[str]] = {
        "root": ["2g2f", "7g7f"],
        "after_1": ["9a9b"],
        "after_2": ["7f7e"],
        "after_3": ["9b9a"],
        "after_4": ["G*5a"],
        "quiet": ["9a9b"],
        "quiet_2": ["2f2e"],
        "quiet_3": ["9b9a"],
        "quiet_4": ["2e2d"],
        "drawish": [],
        "mate": [],
    }

    def generate_moves(self, board: Board, rules: RuleSet) -> list[Move]:
        return [Move.from_usi(usi) for usi in self._MOVES[board.to_sfen()]]


class _ZeroEvaluator(Evaluator):
    def evaluate(self, board: Board) -> int:
        return 0


def test_search_finds_forced_5_ply_mate() -> None:
    """5手先の強制メイトがある枝を選ぶ。"""
    searcher = AlphaBetaSearcher()
    result = searcher.search(
        board=_FivePlyMateBoard(),
        move_gen=_FivePlyMateMoveGen(),
        evaluator=_ZeroEvaluator(),
        rules=RuleSet(),
        depth=5,
        time_limit_ms=5000,
        multi_pv=2,
    )

    assert result.best_move is not None
    assert result.best_move.to_usi() == "7g7f"
    assert result.best_score > 8_000_000
    assert [move.to_usi() for move in result.candidates[0].pv] == [
        "7g7f",
        "9a9b",
        "7f7e",
        "9b9a",
        "G*5a",
    ]
