from __future__ import annotations

import time

from core.board import Board
from core.move_gen import MoveGenerator
from core.rules import RuleSet
from core.types import PIECE_VALUES, Move, PieceType
from eval.base import Evaluator
from search.base import CandidateMove, SearchResult, Searcher
from search.tt import TTEntry, TTFlag, TranspositionTable

_INF = 10_000_000
_MATE_SCORE = 9_000_000
_NULL_MOVE_REDUCTION = 3   # Null move 深さ削減量
_QSEARCH_MAX_DEPTH = 10    # 静止探索の最大深さ


class AlphaBetaSearcher(Searcher):
    """Alpha-beta + 反復深化 + MultiPV。Phase 3 機能:
    - Move Ordering (MVV-LVA + Killer Move)
    - Quiescence Search (静止探索)
    - Null Move Pruning
    """

    def __init__(self) -> None:
        self._tt = TranspositionTable()
        self._nodes = 0
        self._stop = False
        self._start_time = 0.0
        self._time_limit_ms = 3000
        self._killers: dict[int, list[Move]] = {}
        # 探索中に外から参照できるよう属性として保持
        self._move_gen: MoveGenerator | None = None
        self._evaluator: Evaluator | None = None
        self._rules: RuleSet | None = None

    # ------------------------------------------------------------------ public

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
        self._nodes = 0
        self._stop = False
        self._start_time = time.time()
        self._time_limit_ms = time_limit_ms
        self._move_gen = move_gen
        self._evaluator = evaluator
        self._rules = rules
        self._killers.clear()

        best_result: SearchResult | None = None

        for d in range(1, depth + 1):
            if self._is_time_up():
                break
            result = self._search_root(board, d, multi_pv)
            if result is not None:
                if best_result is None or len(result.candidates) >= len(best_result.candidates):
                    best_result = result
            if self._stop:
                break

        if best_result is None:
            moves = move_gen.generate_moves(board, rules)
            if moves:
                best_result = SearchResult(
                    best_move=moves[0],
                    best_score=0,
                    candidates=[CandidateMove(move=moves[0], score=0, pv=[moves[0]])],
                    depth=0,
                    nodes=self._nodes,
                )
            else:
                best_result = SearchResult(
                    best_move=None,
                    best_score=-_MATE_SCORE,
                    candidates=[],
                    depth=0,
                    nodes=self._nodes,
                )

        return best_result

    def stop(self) -> None:
        self._stop = True

    # --------------------------------------------------------------- search root

    def _search_root(self, board: Board, depth: int, multi_pv: int) -> SearchResult | None:
        assert self._move_gen is not None
        assert self._rules is not None
        moves = self._move_gen.generate_moves(board, self._rules)
        if not moves:
            return None

        # TT 最善手でルート手をソートしてから MultiPV 探索
        tt_best = None
        tt_entry = self._tt.get(board.to_sfen())
        if tt_entry:
            tt_best = tt_entry.best_move
        ordered_moves = self._order_moves(board, moves, depth, tt_best)

        scored: list[tuple[int, Move, list[Move]]] = []
        excluded: set[str] = set()

        for _ in range(min(multi_pv, len(ordered_moves))):
            if self._is_time_up() or self._stop:
                break

            best_score = -_INF
            best_move: Move | None = None
            pv: list[Move] = []

            for move in ordered_moves:
                if move.to_usi() in excluded:
                    continue
                new_board = board.apply_move(move)
                child_pv: list[Move] = []
                score = -self._alphabeta(new_board, depth - 1, -_INF, _INF, child_pv)
                if score > best_score:
                    best_score = score
                    best_move = move
                    pv = [move] + child_pv

            if best_score == -_INF or best_move is None:
                break

            scored.append((best_score, best_move, pv))
            excluded.add(best_move.to_usi())

        if not scored:
            return None

        candidates = [CandidateMove(move=m, score=s, pv=p) for s, m, p in scored]
        return SearchResult(
            best_move=candidates[0].move,
            best_score=candidates[0].score,
            candidates=candidates,
            depth=depth,
            nodes=self._nodes,
        )

    # --------------------------------------------------------------- alphabeta

    def _alphabeta(
        self,
        board: Board,
        depth: int,
        alpha: int,
        beta: int,
        pv: list[Move],
        is_null_move: bool = False,
    ) -> int:
        assert self._move_gen is not None
        assert self._evaluator is not None
        assert self._rules is not None
        self._nodes += 1

        if board.is_game_over():
            return -(_MATE_SCORE - depth)

        if depth == 0:
            return self._quiescence(board, alpha, beta, _QSEARCH_MAX_DEPTH)

        if self._is_time_up() or self._stop:
            return self._evaluator.evaluate(board)

        # TT lookup
        key = board.to_sfen()
        tt_entry = self._tt.get(key)
        tt_best_move: Move | None = tt_entry.best_move if tt_entry else None
        if tt_entry and tt_entry.depth >= depth:
            if tt_entry.flag == TTFlag.EXACT:
                return tt_entry.score
            elif tt_entry.flag == TTFlag.LOWER:
                alpha = max(alpha, tt_entry.score)
            elif tt_entry.flag == TTFlag.UPPER:
                beta = min(beta, tt_entry.score)
            if alpha >= beta:
                return tt_entry.score

        # Null Move Pruning
        if not is_null_move and depth >= _NULL_MOVE_REDUCTION and not board.is_check():
            null_board = board.null_move_board()
            null_score = -self._alphabeta(
                null_board, depth - _NULL_MOVE_REDUCTION, -beta, -beta + 1, [], True
            )
            if null_score >= beta:
                return beta

        moves = self._move_gen.generate_moves(board, self._rules)
        if not moves:
            return -(_MATE_SCORE - depth)

        moves = self._order_moves(board, moves, depth, tt_best_move)

        best_score = -_INF
        best_move: Move | None = None

        for move in moves:
            new_board = board.apply_move(move)
            child_pv: list[Move] = []
            score = -self._alphabeta(new_board, depth - 1, -beta, -alpha, child_pv)

            if score > best_score:
                best_score = score
                best_move = move
                pv.clear()
                pv.append(move)
                pv.extend(child_pv)

            alpha = max(alpha, score)
            if alpha >= beta:
                if board.piece_at_sq(move.to_sq) is None:
                    self._add_killer(depth, move)
                self._tt.put(key, TTEntry(depth=depth, score=best_score, flag=TTFlag.LOWER, best_move=best_move))
                return best_score

        flag = TTFlag.EXACT if best_score > -_INF else TTFlag.UPPER
        self._tt.put(key, TTEntry(depth=depth, score=best_score, flag=flag, best_move=best_move))
        return best_score

    # ----------------------------------------------------------- quiescence

    def _quiescence(self, board: Board, alpha: int, beta: int, qdepth: int) -> int:
        assert self._move_gen is not None
        assert self._evaluator is not None
        assert self._rules is not None
        self._nodes += 1

        stand_pat = self._evaluator.evaluate(board)
        if stand_pat >= beta:
            return beta
        alpha = max(alpha, stand_pat)

        if board.is_game_over() or qdepth <= 0:
            return alpha

        moves = self._move_gen.generate_moves(board, self._rules)
        captures = [m for m in moves if board.piece_at_sq(m.to_sq) is not None]
        captures = self._order_captures(board, captures)

        for move in captures:
            new_board = board.apply_move(move)
            score = -self._quiescence(new_board, -beta, -alpha, qdepth - 1)
            if score >= beta:
                return beta
            alpha = max(alpha, score)

        return alpha

    # ----------------------------------------------------------- move ordering

    def _order_moves(
        self, board: Board, moves: list[Move], depth: int, tt_best: Move | None
    ) -> list[Move]:
        killers = self._killers.get(depth, [])

        def priority(move: Move) -> int:
            if tt_best and move == tt_best:
                return 10000

            victim_info = board.piece_at_sq(move.to_sq)
            if victim_info is not None:
                victim_value = PIECE_VALUES.get(victim_info[0], 0)
                if move.from_sq is not None:
                    att_info = board.piece_at_sq(move.from_sq)
                    att_value = PIECE_VALUES.get(att_info[0], 0) if att_info else 0
                else:
                    att_value = PIECE_VALUES.get(move.drop_piece_type, 0) if move.drop_piece_type else 0
                return 8000 + victim_value - att_value // 100

            if move in killers:
                return 3000 - killers.index(move) * 100

            if move.promote:
                return 4000

            if move.from_sq is None:
                return 2000

            return 0

        return sorted(moves, key=priority, reverse=True)

    def _order_captures(self, board: Board, captures: list[Move]) -> list[Move]:
        def priority(move: Move) -> int:
            victim_info = board.piece_at_sq(move.to_sq)
            if victim_info is None:
                return 0
            victim_value = PIECE_VALUES.get(victim_info[0], 0)
            if move.from_sq is not None:
                att_info = board.piece_at_sq(move.from_sq)
                att_value = PIECE_VALUES.get(att_info[0], 0) if att_info else 0
            else:
                att_value = 0
            return victim_value - att_value // 100

        return sorted(captures, key=priority, reverse=True)

    def _add_killer(self, depth: int, move: Move) -> None:
        killers = self._killers.setdefault(depth, [])
        if move not in killers:
            killers.insert(0, move)
            if len(killers) > 2:
                killers.pop()

    def _is_time_up(self) -> bool:
        return (time.time() - self._start_time) * 1000 >= self._time_limit_ms
