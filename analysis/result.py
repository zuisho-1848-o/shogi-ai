"""SearchResult を表示向けに整形するユーティリティ。"""
from __future__ import annotations

from search.base import SearchResult


def format_candidates(
    result: SearchResult,
    *,
    max_n: int = 5,
    black_turn: bool = True,
) -> list[dict]:
    """SearchResult の候補手リストを JSON シリアライズ可能な dict リストに変換する。

    スコアは常に先手視点（正=先手有利）に正規化する。
    black_turn=False（後手番）のときは符号を反転する。
    """
    sign = 1 if black_turn else -1
    return [
        {
            "rank": i + 1,
            "move": c.move.to_usi(),
            "score": c.score * sign,
            "pv": [m.to_usi() for m in c.pv],
        }
        for i, c in enumerate(result.candidates[:max_n])
    ]
