from __future__ import annotations

import argparse
import time
from pathlib import Path

import shogi

from book.sfen_book import SfenBook
from book.strategy import STRATEGY_MAP, Strategy
from core.board import Board, PythonShogiBoard
from core.move_gen import PythonShogiMoveGen
from core.rules import RuleSet
from core.types import Color, Move
from engine.config import EngineConfig
from eval.material import MaterialEvaluator
from eval.pst import PSTEvaluator
from search.alphabeta import AlphaBetaSearcher

_PIECE_LABELS: dict[int, str] = {
    shogi.PAWN: "歩",
    shogi.LANCE: "香",
    shogi.KNIGHT: "桂",
    shogi.SILVER: "銀",
    shogi.GOLD: "金",
    shogi.BISHOP: "角",
    shogi.ROOK: "飛",
    shogi.KING: "玉",
    shogi.PROM_PAWN: "と",
    shogi.PROM_LANCE: "杏",
    shogi.PROM_KNIGHT: "圭",
    shogi.PROM_SILVER: "全",
    shogi.PROM_BISHOP: "馬",
    shogi.PROM_ROOK: "龍",
}


def _piece_label(piece: shogi.Piece | None) -> str:
    if piece is None:
        return " . "
    label = _PIECE_LABELS.get(piece.piece_type, "?")
    return f" {label} " if piece.color == shogi.BLACK else f"v{label} "


def _hand_text(board: shogi.Board, color: int) -> str:
    pieces = []
    for piece_type in (shogi.ROOK, shogi.BISHOP, shogi.GOLD, shogi.SILVER, shogi.KNIGHT, shogi.LANCE, shogi.PAWN):
        count = board.pieces_in_hand[color].get(piece_type, 0)
        if count:
            suffix = "" if count == 1 else str(count)
            pieces.append(f"{_PIECE_LABELS[piece_type]}{suffix}")
    return " ".join(pieces) if pieces else "-"


def render_board(board: Board, last_move: Move | None = None) -> str:
    assert isinstance(board, PythonShogiBoard)
    shogi_board = board.get_shogi_board()
    last = last_move.to_usi() if last_move is not None else "-"
    side = "先手" if board.turn == Color.BLACK else "後手"

    lines = [
        f"手番: {side}    last: {last}",
        f"後手 持ち駒: {_hand_text(shogi_board, shogi.WHITE)}",
        "      9   8   7   6   5   4   3   2   1",
        "    +---+---+---+---+---+---+---+---+---+",
    ]
    for rank in range(1, 10):
        cells = []
        for file in range(9, 0, -1):
            sq = shogi.SQUARE_NAMES.index(f"{file}{chr(ord('a') + rank - 1)}")
            cells.append(_piece_label(shogi_board.piece_at(sq)))
        rank_char = chr(ord("a") + rank - 1)
        lines.append(f" {rank_char}  |" + "|".join(cells) + "|")
        lines.append("    +---+---+---+---+---+---+---+---+---+")
    lines.append(f"先手 持ち駒: {_hand_text(shogi_board, shogi.BLACK)}")
    return "\n".join(lines)


def _build_evaluator(config: EngineConfig) -> MaterialEvaluator | PSTEvaluator:
    if config.eval == "pst":
        return PSTEvaluator()
    return MaterialEvaluator()


def _load_book(path_text: str) -> SfenBook | None:
    if path_text.lower() == "none":
        return None
    path = Path(path_text)
    if path.exists():
        return SfenBook.from_file(path)
    return SfenBook.minimal()


def _pick_move(
    board: Board,
    config: EngineConfig,
    strategy: Strategy | None,
    book: SfenBook | None,
) -> tuple[Move | None, str]:
    if book is not None:
        tag = strategy.tag if strategy is not None else None
        book_move = book.lookup(board.to_sfen(), tag)
        if book_move is not None:
            return book_move, "book"

    searcher = AlphaBetaSearcher()
    result = searcher.search(
        board=board,
        move_gen=PythonShogiMoveGen(),
        evaluator=_build_evaluator(config),
        rules=RuleSet(),
        depth=config.depth,
        time_limit_ms=config.time_limit_ms,
        multi_pv=config.multi_pv,
    )
    return result.best_move, f"search depth={result.depth} score={result.best_score} nodes={result.nodes}"


def main() -> None:
    parser = argparse.ArgumentParser(description="CLI board viewer for shogi-ai")
    parser.add_argument("--sfen", default=None, help="開始局面 SFEN。省略時は初期局面")
    parser.add_argument("--plies", type=int, default=8, help="表示しながら進める手数")
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--time-limit-ms", type=int, default=1000)
    parser.add_argument("--multi-pv", type=int, default=3)
    parser.add_argument("--eval", default="pst", choices=["material", "pst"])
    parser.add_argument("--book", default="book/standard.sfen", help="定跡ファイルパス（none で無効）")
    parser.add_argument("--strategy", default=None, choices=list(STRATEGY_MAP.keys()))
    parser.add_argument("--delay", type=float, default=0.0, help="各手の表示後に待つ秒数")
    parser.add_argument("--clear", action="store_true", help="各手ごとに画面をクリアして表示")
    args = parser.parse_args()

    config = EngineConfig(
        eval=args.eval,
        depth=args.depth,
        time_limit_ms=args.time_limit_ms,
        multi_pv=args.multi_pv,
    )
    board: Board = PythonShogiBoard.from_sfen(args.sfen) if args.sfen else PythonShogiBoard.initial()
    book = _load_book(args.book)
    strategy = STRATEGY_MAP.get(args.strategy) if args.strategy else None

    last_move: Move | None = None
    for ply in range(args.plies + 1):
        if args.clear:
            print("\033[2J\033[H", end="")
        print(f"\n=== ply {ply} ===")
        print(render_board(board, last_move))

        if ply == args.plies or board.is_game_over():
            break

        move, source = _pick_move(board, config, strategy, book)
        if move is None:
            print("bestmove resign")
            break
        print(f"\nAI: {move.to_usi()} ({source})")
        board = board.apply_move(move)
        last_move = move
        if args.delay > 0:
            time.sleep(args.delay)


if __name__ == "__main__":
    main()
