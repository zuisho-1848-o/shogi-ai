from __future__ import annotations

from pathlib import Path

from book.base import OpeningBook
from book.strategy import Strategy
from core.types import Move


class BookEntry:
    __slots__ = ("move", "score", "tag")

    def __init__(self, move: Move, score: int, tag: str | None) -> None:
        self.move = move
        self.score = score
        self.tag = tag


class SfenBook(OpeningBook):
    """SFEN テキスト形式の定跡ファイルを読み込む実装。

    形式:
        # コメント
        sfen <board> <turn> <hands>
        <move> <score> [<tag>]
        ...
    """

    def __init__(self, entries: dict[str, list[BookEntry]] | None = None) -> None:
        self._book: dict[str, list[BookEntry]] = entries or {}

    # ----------------------------------------------------------------- public

    def lookup(self, sfen: str, strategy_tag: str | None = None) -> Move | None:
        key = self.normalize_sfen(sfen)
        entries = self._book.get(key)
        if not entries:
            return None

        if strategy_tag is not None:
            # タグ一致の手を優先、なければ全体から最高スコア
            tagged = [e for e in entries if e.tag == strategy_tag]
            pool = tagged if tagged else entries
        else:
            pool = entries

        return max(pool, key=lambda e: e.score).move

    @classmethod
    def from_file(cls, path: Path) -> SfenBook:
        book: dict[str, list[BookEntry]] = {}
        current_key: str | None = None

        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("sfen "):
                    parts = line.split(maxsplit=1)
                    current_key = parts[1].strip() if len(parts) > 1 else None
                    if current_key and current_key not in book:
                        book[current_key] = []
                elif current_key is not None:
                    tokens = line.split()
                    if not tokens:
                        continue
                    move_usi = tokens[0]
                    score = int(tokens[1]) if len(tokens) >= 2 else 0
                    tag = tokens[2] if len(tokens) >= 3 else None
                    try:
                        move = Move.from_usi(move_usi)
                        book[current_key].append(BookEntry(move, score, tag))
                    except (ValueError, KeyError):
                        pass

        return cls(book)

    @classmethod
    def minimal(cls) -> SfenBook:
        """ファイルなしで使える最小定跡（テスト・デバッグ用）。"""
        _INITIAL = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b -"
        entries = {
            _INITIAL: [
                BookEntry(Move.from_usi("7g7f"), 100, "static_rook"),
                BookEntry(Move.from_usi("2g2f"), 90, "static_rook"),
                BookEntry(Move.from_usi("7g7f"), 100, "ranging_rook"),
                BookEntry(Move.from_usi("6g6f"), 80, None),
            ]
        }
        return cls(entries)
