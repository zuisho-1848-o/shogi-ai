"""棋譜の全局面を解析し、1手ごとの評価値・最善手・損失を求める。"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import shogi

from .engine import MATE_CP, Line, UsiEngine
from .kif import KifGame


def winrate(cp: float) -> float:
    """評価値(手番側視点) → 勝率。ShogiGUI等で一般的な 600 スケール。"""
    return 1.0 / (1.0 + math.exp(-max(-3000, min(3000, cp)) / 600.0))


@dataclass
class PosAnalysis:
    """1局面（n手目を指す直前）の解析結果。手番側視点。"""

    ply: int  # 0 = 開始局面
    lines: list[Line] = field(default_factory=list)  # MultiPV（良い順）
    terminal: bool = False  # 詰み等で合法手なし

    @property
    def best(self) -> Line | None:
        return self.lines[0] if self.lines else None

    @property
    def cp(self) -> int:
        """手番側視点の評価値。終局（詰まされ）は大きな負値。"""
        return self.best.score_cp if self.best else -MATE_CP


def cache_key(game: KifGame, engine_name: str, movetime: int, depth: int | None, multipv: int, eval_id: str) -> str:
    h = hashlib.sha1()
    h.update(json.dumps([game.start_sfen, [m.usi for m in game.moves], engine_name, movetime, depth, multipv, eval_id]).encode())
    return h.hexdigest()[:16]


def analyze_game(
    game: KifGame,
    engine: UsiEngine,
    movetime_ms: int = 1000,
    depth: int | None = None,
    progress=None,
    cache_path: Path | None = None,
    key: str = "",
) -> list[PosAnalysis]:
    """局面 0..N（N=総手数）を解析。キャッシュがあれば再利用して中断再開できる。"""
    n = len(game.moves)
    results: dict[int, PosAnalysis] = {}
    if cache_path and cache_path.exists():
        data = json.loads(cache_path.read_text())
        if data.get("key") == key:
            for ply, d in data["positions"].items():
                results[int(ply)] = PosAnalysis(
                    int(ply), [Line(l["multipv"], l["cp"], l["mate"], l["depth"], l["pv"]) for l in d["lines"]], d["terminal"]
                )
    board = shogi.Board(game.start_sfen)
    usi_moves = [m.usi for m in game.moves]
    for ply in range(n + 1):
        if ply not in results:
            if board.is_game_over():
                results[ply] = PosAnalysis(ply, [], True)
            else:
                lines = engine.analyze(game.start_sfen, usi_moves[:ply], movetime_ms, depth)
                results[ply] = PosAnalysis(ply, lines, not lines)
            if cache_path:
                cache_path.write_text(
                    json.dumps(
                        {"key": key, "positions": {str(p): {"lines": [l.to_dict() for l in a.lines], "terminal": a.terminal} for p, a in results.items()}},
                        ensure_ascii=False,
                    )
                )
        if progress:
            progress(ply, n)
        if ply < n:
            board.push_usi(usi_moves[ply])
    return [results[i] for i in range(n + 1)]


def eval_id_of(path: Path) -> str:
    """評価関数ファイルの識別子（サイズ+更新時刻）。キャッシュ無効化用。"""
    try:
        st = path.stat()
        return f"{st.st_size}-{int(st.st_mtime)}"
    except OSError:
        return "none"
