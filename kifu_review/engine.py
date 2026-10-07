"""USIエンジン（やねうら王）の最小クライアント。MultiPV解析に特化。"""

from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

MATE_CP = 30000  # 詰みスコアを cp 換算する基準値（詰み手数分だけ引く）


@dataclass
class Line:
    """MultiPV の1本分。score_cp は『手番側から見た』評価値。"""

    multipv: int
    score_cp: int
    mate: int | None  # 手番側が N手詰(+)/詰まされる(-)。詰みでなければ None
    depth: int
    pv: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"multipv": self.multipv, "cp": self.score_cp, "mate": self.mate, "depth": self.depth, "pv": self.pv}


def mate_to_cp(mate: int) -> int:
    return (MATE_CP - abs(mate)) * (1 if mate > 0 else -1)


_INFO_RE = re.compile(r"\bmultipv (\d+)")
_SCORE_RE = re.compile(r"\bscore (cp|mate) (-?\d+|\+|-)(?: (lowerbound|upperbound))?")


class UsiEngine:
    def __init__(
        self,
        path: str | Path,
        eval_dir: str | Path | None = None,
        threads: int = 4,
        hash_mb: int = 512,
        multipv: int = 3,
        fv_scale: int | None = None,
    ):
        self.path = Path(path).expanduser().resolve()
        if not self.path.exists():
            raise FileNotFoundError(f"エンジンが見つかりません: {self.path}")
        eval_dir = Path(eval_dir).expanduser().resolve() if eval_dir else self.path.parent / "eval"
        self.proc = subprocess.Popen(
            [str(self.path)], cwd=self.path.parent, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1,
        )  # fmt: skip
        self.name = ""
        self._send("usi")
        self._wait("usiok", on_line=self._grab_name)
        for k, v in {
            "EvalDir": str(eval_dir), "Threads": threads, "USI_Hash": hash_mb,
            "MultiPV": multipv, "BookFile": "no_book", "NetworkDelay": 0, "NetworkDelay2": 0,
            "MinimumThinkingTime": 0,
        }.items():  # fmt: skip
            self._send(f"setoption name {k} value {v}")
        if fv_scale:
            self._send(f"setoption name FV_SCALE value {fv_scale}")
        self._send("isready")
        self._wait("readyok")
        self._send("usinewgame")
        self.multipv = multipv

    def _send(self, cmd: str) -> None:
        assert self.proc.stdin
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def _grab_name(self, line: str) -> None:
        if line.startswith("id name "):
            self.name = line[8:].strip()

    def _wait(self, token: str, on_line=None, timeout: float = 120.0) -> None:
        assert self.proc.stdout
        t0 = time.time()
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("エンジンが予期せず終了しました")
            line = line.strip()
            if on_line:
                on_line(line)
            if line == token or line.startswith(token + " "):
                return
            if time.time() - t0 > timeout:
                raise TimeoutError(f"エンジンが {token} を返しません")

    def analyze(self, start_sfen: str, moves: list[str], movetime_ms: int = 1000, depth: int | None = None) -> list[Line]:
        """局面（開始局面＋指し手列）を解析し、MultiPV の各ラインを返す。"""
        pos = f"position sfen {start_sfen}" + (" moves " + " ".join(moves) if moves else "")
        self._send(pos)
        self._send(f"go depth {depth}" if depth else f"go movetime {movetime_ms}")
        best: dict[int, tuple[bool, Line]] = {}  # multipv -> (is_exact, line)
        assert self.proc.stdout
        while True:
            raw = self.proc.stdout.readline()
            if not raw:
                raise RuntimeError("エンジンが予期せず終了しました")
            line = raw.strip()
            if line.startswith("bestmove"):
                if not best:  # 合法手なし等
                    return []
                return [v[1] for _, v in sorted(best.items())]
            if not line.startswith("info") or " pv " not in line:
                continue
            sm = _SCORE_RE.search(line)
            if not sm:
                continue
            mp = _INFO_RE.search(line)
            idx = int(mp.group(1)) if mp else 1
            kind, val, bound = sm.groups()
            if val in ("+", "-"):
                mate = 1 if val == "+" else -1
                cp = mate_to_cp(mate)
            elif kind == "mate":
                mate, cp = int(val), mate_to_cp(int(val))
            else:
                mate, cp = None, int(val)
            dm = re.search(r"\bdepth (\d+)", line)
            pv = line.split(" pv ", 1)[1].split()
            cand = Line(idx, cp, mate, int(dm.group(1)) if dm else 0, pv)
            exact = bound is None
            prev = best.get(idx)
            # 境界付き（lower/upper）は、確定値がまだ無い場合にのみ採用
            if prev is None or exact or not prev[0]:
                best[idx] = (exact or (prev[0] if prev else False), cand)

    def close(self) -> None:
        try:
            self._send("quit")
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()

    def __enter__(self) -> "UsiEngine":
        return self

    def __exit__(self, *a) -> None:
        self.close()
