from __future__ import annotations

from dataclasses import dataclass, field

from core.board import Board, PythonShogiBoard
from core.move_gen import PythonShogiMoveGen
from core.rules import RuleSet
from core.types import Move
from solve.db.confirmed import ConfirmedDB, Label

_move_gen = PythonShogiMoveGen()
_rules = RuleSet()


@dataclass
class SolveStats:
    nodes_visited: int = 0
    confirmed_win: int = 0
    confirmed_loss: int = 0
    cache_hits: int = 0
    max_depth_reached: int = 0
    # 各深さで確定できた局面数
    confirmed_by_depth: dict[int, int] = field(default_factory=dict)


def solve(
    board: Board,
    db: ConfirmedDB,
    depth_limit: int = 10,
    *,
    fast: bool = False,
    _depth: int = 0,
    _stats: SolveStats | None = None,
    _stack: set[str] | None = None,
) -> Label | None:
    """fast=True: 勝ち手を1つ見つけた時点で探索を打ち切る（高速だが best_moves が不完全）。
    fast=False（デフォルト）: 全手を探索し、DBに全勝ち局面を登録する。
    """
    """局面を再帰的に解く。

    Returns:
        Label.WIN  … 手番側が最善手を打てば勝てる
        Label.LOSS … 手番側が最善手を打っても負ける
        None       … depth_limit 内では確定できなかった
    """
    if _stats is None:
        _stats = SolveStats()
    if _stack is None:
        _stack = set()

    sfen = board.to_sfen()
    key = " ".join(sfen.split()[:3])

    # --- キャッシュヒット ---
    cached = db.get(sfen)
    if cached is not None:
        _stats.cache_hits += 1
        return cached

    # --- 千日手検出（スタック上に同一局面） ---
    if key in _stack:
        # 千日手はひとまず None（不明）として返す
        return None

    _stats.nodes_visited += 1

    # --- 終端局面（合法手なし） ---
    if board.is_game_over():
        # 将棋のルール上、合法手がない = 詰み or 入玉禁止等 → 手番側の負け
        label = Label.LOSS
        db.set(sfen, label)
        _stats.confirmed_loss += 1
        _stats.confirmed_by_depth[_depth] = _stats.confirmed_by_depth.get(_depth, 0) + 1
        return label

    # --- 深さ上限 ---
    if _depth >= depth_limit:
        _stats.max_depth_reached += 1
        return None

    moves = _move_gen.generate_moves(board, _rules)
    if not moves:
        label = Label.LOSS
        db.set(sfen, label)
        _stats.confirmed_loss += 1
        return label

    _stack.add(key)

    has_unknown = False
    best: Label | None = None

    for move in moves:
        next_board = board.apply_move(move)
        result = solve(next_board, db, depth_limit, fast=fast, _depth=_depth + 1, _stats=_stats, _stack=_stack)

        if result == Label.LOSS:
            # 相手が詰み（負け）になる手がある → 自分は勝ち確定
            best = Label.WIN
            if fast:
                break  # 高速モード: 1手見つけたら即終了
        elif result is None:
            has_unknown = True
        # result == Label.WIN → 相手が勝てる手 → 自分には不利

    _stack.discard(key)

    if best == Label.WIN:
        db.set(sfen, Label.WIN)
        _stats.confirmed_win += 1
        _stats.confirmed_by_depth[_depth] = _stats.confirmed_by_depth.get(_depth, 0) + 1
        return Label.WIN

    if not has_unknown:
        # 全ての手が Label.WIN（相手有利）→ 自分はどこに指しても負け
        db.set(sfen, Label.LOSS)
        _stats.confirmed_loss += 1
        _stats.confirmed_by_depth[_depth] = _stats.confirmed_by_depth.get(_depth, 0) + 1
        return Label.LOSS

    return None


def solve_with_stats(
    board: Board,
    db: ConfirmedDB,
    depth_limit: int = 10,
    fast: bool = False,
) -> tuple[Label | None, SolveStats]:
    """solve() を実行してラベルと統計情報を返す。"""
    stats = SolveStats()
    label = solve(board, db, depth_limit, fast=fast, _stats=stats)
    return label, stats


def best_moves(board: Board, db: ConfirmedDB) -> list[Move]:
    """DBを参照してこの局面の最善手リストを返す（WINを導く手を優先）。

    DBに未登録の手は含めない。
    """
    moves = _move_gen.generate_moves(board, _rules)
    winning: list[Move] = []
    for move in moves:
        next_board = board.apply_move(move)
        label = db.get(next_board.to_sfen())
        if label == Label.LOSS:
            # 相手が負け局面に追い込める → 勝ち手
            winning.append(move)
    return winning
