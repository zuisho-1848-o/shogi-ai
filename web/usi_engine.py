"""USI プロトコル対応の外部将棋エンジンラッパー（非同期版）"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class USICandidate:
    multipv: int
    move: str
    score: int
    pv: list[str] = field(default_factory=list)


@dataclass
class USISearchResult:
    best_move: str | None
    score: int
    candidates: list[dict]


class USIEngine:
    """USI エンジンのサブプロセスを管理し、探索結果を返す。"""

    def __init__(self, path: str) -> None:
        self.path = path
        self.name: str = Path(path).name
        self._proc: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        self._proc = await asyncio.create_subprocess_exec(
            self.path,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await self._send("usi")
        async for line in self._read_until("usiok"):
            if line.startswith("id name "):
                self.name = line[8:].strip()

        # 定跡なしにする（ファイル未同梱でもエラーにならないよう）
        await self._send("setoption name BookFile value no_book")
        await self._send("isready")
        async for _ in self._read_until("readyok"):
            pass

    async def new_game(self) -> None:
        await self._send("usinewgame")

    async def search(
        self,
        moves: list[str],
        time_limit_ms: int = 5000,
        multi_pv: int = 5,
    ) -> USISearchResult:
        async with self._lock:
            pos_cmd = "position startpos"
            if moves:
                pos_cmd += " moves " + " ".join(moves)
            await self._send(f"setoption name MultiPV value {multi_pv}")
            await self._send(pos_cmd)
            await self._send(f"go movetime {time_limit_ms}")

            cand_map: dict[int, USICandidate] = {}
            best_move: str | None = None

            async for line in self._read_until("bestmove"):
                if line.startswith("bestmove"):
                    parts = line.split()
                    bm = parts[1] if len(parts) > 1 else "resign"
                    best_move = None if bm in ("resign", "win") else bm
                    break
                if line.startswith("info ") and "score" in line and "pv" in line:
                    cand = _parse_info(line)
                    if cand:
                        cand_map[cand.multipv] = cand

            candidates = _format_candidates(cand_map)
            score = candidates[0]["score"] if candidates else 0
            return USISearchResult(best_move=best_move, score=score, candidates=candidates)

    async def quit(self) -> None:
        if self._proc is None:
            return
        try:
            await self._send("quit")
            await asyncio.wait_for(self._proc.wait(), timeout=3.0)
        except Exception:
            pass
        finally:
            if self._proc and self._proc.returncode is None:
                self._proc.terminate()
            self._proc = None

    @property
    def is_running(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    async def _send(self, cmd: str) -> None:
        if self._proc and self._proc.stdin:
            self._proc.stdin.write((cmd + "\n").encode())
            await self._proc.stdin.drain()

    async def _read_until(self, sentinel: str):
        """sentinel を含む行まで行を yield する。"""
        if not self._proc or not self._proc.stdout:
            return
        while True:
            try:
                raw = await asyncio.wait_for(self._proc.stdout.readline(), timeout=60.0)
            except asyncio.TimeoutError:
                break
            if not raw:
                break
            line = raw.decode().strip()
            yield line
            if line.startswith(sentinel):
                break


def _parse_info(line: str) -> USICandidate | None:
    parts = line.split()
    info: dict = {}
    i = 1
    while i < len(parts):
        key = parts[i]
        if key == "multipv" and i + 1 < len(parts):
            info["multipv"] = int(parts[i + 1]); i += 2
        elif key == "depth" and i + 1 < len(parts):
            i += 2
        elif key == "score" and i + 2 < len(parts):
            stype, sval = parts[i + 1], parts[i + 2]
            if stype == "cp":
                info["score"] = int(sval)
            elif stype == "mate":
                v = int(sval)
                info["score"] = 30000 if v > 0 else -30000
            i += 3
            # upperbound/lowerbound 読み飛ばし
            if i < len(parts) and parts[i] in ("upperbound", "lowerbound"):
                i += 1
        elif key == "pv" and i + 1 < len(parts):
            info["pv"] = parts[i + 1:]
            info["move"] = parts[i + 1]
            break
        else:
            i += 1
    if "move" not in info:
        return None
    return USICandidate(
        multipv=info.get("multipv", 1),
        move=info["move"],
        score=info.get("score", 0),
        pv=info.get("pv", []),
    )


def _format_candidates(cand_map: dict[int, USICandidate]) -> list[dict]:
    result = []
    for idx in sorted(cand_map):
        c = cand_map[idx]
        result.append({
            "rank": idx,
            "move": c.move,
            "score": c.score,
            "pv": c.pv,
        })
    return result
