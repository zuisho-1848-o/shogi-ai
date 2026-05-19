"""SearchResult を表示向けに整形するユーティリティ。"""
from __future__ import annotations

from search.base import SearchResult


def format_candidates(result: SearchResult, *, max_n: int = 5) -> list[dict]:
    """SearchResult の候補手リストを JSON シリアライズ可能な dict リストに変換する。"""
    return [
        {
            "rank": i + 1,
            "move": c.move.to_usi(),
            "score": c.score,
            "pv": [m.to_usi() for m in c.pv],
        }
        for i, c in enumerate(result.candidates[:max_n])
    ]
