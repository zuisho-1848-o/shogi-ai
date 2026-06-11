from __future__ import annotations

import json
import os
from enum import Enum


class Label(Enum):
    WIN = "win"    # 手番側が最善手を打てば勝てる
    LOSS = "loss"  # 手番側が最善手を打っても負ける
    DRAW = "draw"  # 引き分け（千日手・持将棋）


class ConfirmedDB:
    """確定ラベルDB。局面ハッシュ（SFEN正規化）→ Label を管理する。

    SFEN の手数カウンタは局面同一性に無関係なので除去して管理する。
    """

    def __init__(self, path: str | None = None) -> None:
        self._db: dict[str, Label] = {}
        self._path = path
        if path and os.path.exists(path):
            self._load()

    # --- 公開API ---

    def get(self, sfen: str) -> Label | None:
        return self._db.get(_normalize(sfen))

    def set(self, sfen: str, label: Label) -> None:
        self._db[_normalize(sfen)] = label

    def __contains__(self, sfen: str) -> bool:
        return _normalize(sfen) in self._db

    def __len__(self) -> int:
        return len(self._db)

    def stats(self) -> dict[str, int]:
        counts: dict[str, int] = {label.value: 0 for label in Label}
        for lbl in self._db.values():
            counts[lbl.value] += 1
        return counts

    def save(self) -> None:
        if not self._path:
            return
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump({k: v.value for k, v in self._db.items()}, f, ensure_ascii=False, indent=2)

    def _load(self) -> None:
        with open(self._path, encoding="utf-8") as f:
            data = json.load(f)
        self._db = {k: Label(v) for k, v in data.items()}


def _normalize(sfen: str) -> str:
    """手数カウンタを除いた局面キーを返す（先頭3フィールド）。"""
    return " ".join(sfen.split()[:3])
