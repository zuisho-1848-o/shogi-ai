from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from core.types import Move


class TTFlag(IntEnum):
    EXACT = 0
    LOWER = 1  # beta cutoff (score is lower bound)
    UPPER = 2  # alpha cutoff (score is upper bound)


@dataclass
class TTEntry:
    depth: int
    score: int
    flag: TTFlag
    best_move: Move | None


class TranspositionTable:
    def __init__(self, max_entries: int = 1 << 20) -> None:
        self._table: dict[str, TTEntry] = {}
        self._max_entries = max_entries

    def get(self, key: str) -> TTEntry | None:
        return self._table.get(key)

    def put(self, key: str, entry: TTEntry) -> None:
        if len(self._table) >= self._max_entries:
            # 古いエントリを半分削除（シンプルな eviction）
            keys = list(self._table.keys())
            for k in keys[: len(keys) // 2]:
                del self._table[k]
        self._table[key] = entry

    def clear(self) -> None:
        self._table.clear()
