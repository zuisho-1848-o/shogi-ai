from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from core.board import Board
from core.move_gen import MoveGenerator
from core.rules import RuleSet
from core.types import Move
from eval.base import Evaluator
from search.base import CandidateMove, SearchResult, Searcher

_UCB_C = math.sqrt(2)
_SCORE_SCALE = 600.0  # centipawns → [0,1] sigmoid の感度調整


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


@dataclass
class _MCTSNode:
    board: Board
    move: Move | None  # このノードへ至った手（root は None）
    parent: _MCTSNode | None
    untried_moves: list[Move]
    children: list[_MCTSNode] = field(default_factory=list)
    visits: int = 0
    # 「このノードへ移動したプレイヤー（親のプレイヤー）」の勝利確率の累積
    total_value: float = 0.0

    def ucb1(self, c: float = _UCB_C) -> float:
        if self.visits == 0:
            return float("inf")
        assert self.parent is not None
        exploit = self.total_value / self.visits
        explore = c * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploit + explore

    def is_fully_expanded(self) -> bool:
        return len(self.untried_moves) == 0

    def best_child(self, c: float = _UCB_C) -> _MCTSNode:
        return max(self.children, key=lambda n: n.ucb1(c))


class MCTSSearcher(Searcher):
    """UCT (Upper Confidence Bound for Trees) MCTS。

    ランダムロールアウトの代わりに評価関数で葉ノードをスコアリングする。
    time_limit_ms で探索時間を制限し、反復回数を最大化する設計。
    """

    def __init__(self, c: float = _UCB_C) -> None:
        self._c = c
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def search(
        self,
        board: Board,
        move_gen: MoveGenerator,
        evaluator: Evaluator,
        rules: RuleSet,
        depth: int,
        time_limit_ms: int,
        multi_pv: int,
    ) -> SearchResult:
        self._stop = False
        deadline = time.time() + time_limit_ms / 1000.0

        root_moves = move_gen.generate_moves(board, rules)
        if not root_moves:
            return SearchResult(
                best_move=None,
                best_score=-9_000_000,
                candidates=[],
                depth=0,
                nodes=0,
            )

        root = _MCTSNode(
            board=board,
            move=None,
            parent=None,
            untried_moves=list(root_moves),
        )

        nodes = 0
        while time.time() < deadline and not self._stop:
            node = self._select(root)
            child = self._expand(node, move_gen, rules)
            value = self._evaluate(child, evaluator)
            self._backpropagate(child, value)
            nodes += 1

        if not root.children:
            # 時間切れで1手も展開できなかった場合のフォールバック
            fallback = root_moves[0]
            return SearchResult(
                best_move=fallback,
                best_score=0,
                candidates=[CandidateMove(move=fallback, score=0, pv=[fallback])],
                depth=0,
                nodes=nodes,
            )

        # 訪問回数でソートして MultiPV 候補を生成
        ranked = sorted(root.children, key=lambda n: n.visits, reverse=True)
        candidates: list[CandidateMove] = []
        for child in ranked[:multi_pv]:
            assert child.move is not None
            win_rate = child.total_value / child.visits if child.visits > 0 else 0.5
            # 勝率 [0,1] → centipawn に逆変換（表示用）。端値をクランプして log(0) を回避
            win_rate = max(1e-6, min(1.0 - 1e-6, win_rate))
            score = int(math.log(win_rate / (1.0 - win_rate)) * _SCORE_SCALE)
            candidates.append(CandidateMove(move=child.move, score=score, pv=[child.move]))

        best = candidates[0]
        return SearchResult(
            best_move=best.move,
            best_score=best.score,
            candidates=candidates,
            depth=0,
            nodes=nodes,
        )

    # ----------------------------------------------------------------- private

    def _select(self, root: _MCTSNode) -> _MCTSNode:
        """完全展開済みのノードを UCB1 で辿り、葉または未展開ノードに到達する。"""
        node = root
        while node.is_fully_expanded() and node.children:
            node = node.best_child(self._c)
        return node

    def _expand(self, node: _MCTSNode, move_gen: MoveGenerator, rules: RuleSet) -> _MCTSNode:
        """未試行の手を1つ取り出して子ノードを追加する。"""
        if not node.untried_moves:
            return node
        move = node.untried_moves.pop()
        new_board = node.board.apply_move(move)
        child_moves = move_gen.generate_moves(new_board, rules) if not new_board.is_game_over() else []
        child = _MCTSNode(
            board=new_board,
            move=move,
            parent=node,
            untried_moves=child_moves,
        )
        node.children.append(child)
        return child

    def _evaluate(self, node: _MCTSNode, evaluator: Evaluator) -> float:
        """ノードの葉評価。戻り値は「このノードへ移動したプレイヤーの勝利確率」[0,1]。"""
        if node.board.is_game_over():
            # 現在の手番プレイヤーが詰まされている = 移動したプレイヤー（親）が勝利
            return 1.0
        score = evaluator.evaluate(node.board)  # 現在の手番側から見た評価
        # 親プレイヤー（移動した側）の視点に反転してから sigmoid
        return _sigmoid(-score / _SCORE_SCALE)

    def _backpropagate(self, node: _MCTSNode, value: float) -> None:
        """value を木の上方向へ伝播。各レベルで視点を反転する。
        node.total_value = 「このノードへ移動したプレイヤー」の勝利確率の累積
        """
        current: _MCTSNode | None = node
        while current is not None:
            current.visits += 1
            current.total_value += value
            value = 1.0 - value  # 親は対戦相手の視点
            current = current.parent
