from __future__ import annotations

from abc import ABC, abstractmethod

from core.types import Move


class OpeningBook(ABC):
    """定跡参照の抽象インターフェース。"""

    @abstractmethod
    def lookup(self, sfen: str, strategy_tag: str | None = None) -> Move | None:
        """正規化 SFEN で定跡手を検索する。見つからなければ None。"""
        ...

    @staticmethod
    def normalize_sfen(sfen: str) -> str:
        """SFEN から手数を除いた正規化形式に変換する。"""
        parts = sfen.split()
        return " ".join(parts[:3])  # board turn hands
