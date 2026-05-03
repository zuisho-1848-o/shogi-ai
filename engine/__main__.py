from __future__ import annotations

import argparse
from pathlib import Path

from book.sfen_book import SfenBook
from book.strategy import STRATEGY_MAP
from engine.config import EngineConfig
from engine.engine import Engine
from engine.usi import run_usi_loop


def main() -> None:
    parser = argparse.ArgumentParser(description="shogi-ai USI engine")
    parser.add_argument("--search", default="alphabeta", choices=["minimax", "alphabeta", "mcts"])
    parser.add_argument("--eval", default="pst", choices=["material", "pst", "kpp", "nnue"])
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--time-limit-ms", type=int, default=3000)
    parser.add_argument("--multi-pv", type=int, default=5)
    parser.add_argument("--book", default="book/standard.sfen", help="定跡ファイルパス（none で無効）")
    parser.add_argument("--strategy", default=None, choices=list(STRATEGY_MAP.keys()) + [None],
                        help="戦法指定（ranging_rook / static_rook / free）")
    args = parser.parse_args()

    config = EngineConfig(
        search=args.search,
        eval=args.eval,
        depth=args.depth,
        time_limit_ms=args.time_limit_ms,
        multi_pv=args.multi_pv,
    )

    # 定跡読み込み
    book = None
    if args.book and args.book.lower() != "none":
        book_path = Path(args.book)
        if book_path.exists():
            book = SfenBook.from_file(book_path)
        else:
            book = SfenBook.minimal()

    strategy = STRATEGY_MAP.get(args.strategy) if args.strategy else None

    engine = Engine(config, book=book, strategy=strategy)
    run_usi_loop(engine)


if __name__ == "__main__":
    main()
