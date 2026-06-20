"""LLMバックエンド抽象化と解説生成"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod

from explain.context_builder import BoardContext

SYSTEM_PROMPTS: dict[str, str] = {
    "beginner": (
        "あなたは将棋の家庭教師AIです。\n"
        "駒の種類・動き方・基本ルールから丁寧に説明し、初心者にも分かりやすい言葉で盤面を解説してください。\n"
        "「王手」「詰み」「持ち駒」などの専門用語は登場時に必ず一言説明を添えてください。\n"
        "返答は必ず日本語でお願いします。"
    ),
    "intermediate": (
        "あなたは将棋の解説AIです。\n"
        "手筋名・定跡名・囲い名を積極的に使いながら、初段前後のプレイヤーに向けて\n"
        "局面の特徴・候補手の意図・想定される展開を解説してください。\n"
        "返答は必ず日本語でお願いします。"
    ),
    "advanced": (
        "あなたはプロ棋士レベルの将棋解説AIです。\n"
        "変化手順・深い読み・戦略的意図を詳細に解説してください。\n"
        "評価値の根拠・定跡の分岐・好手悪手の判断を明確に示してください。\n"
        "返答は必ず日本語でお願いします。"
    ),
}

_USER_TEMPLATE = """\
【現在の局面】
{hands_text}
{board_text}
（先手の駒は通常表示、後手の駒は "v" を頭に付けて表示しています）

【対局情報】
{turn_text}　/ {move_count}手目　/ {phase}{check_line}
{eval_text}

【候補手（上位5件・先手視点の評価値）】
{candidates_text}

以下の観点で日本語で解説してください：
1. 現在の局面の特徴（形勢・駒の配置・陣形）
2. 最善手の狙いと理由
3. この局面から想定される2〜3手先の展開
4. 注意すべき相手の反撃や落とし穴
"""


class LLMBackend(ABC):
    @abstractmethod
    async def chat(self, system: str, user: str) -> str: ...

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def model(self) -> str: ...


class OllamaBackend(LLMBackend):
    def __init__(self, host: str, model: str) -> None:
        self._host = host.rstrip("/")
        self._model = model

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def model(self) -> str:
        return self._model

    async def chat(self, system: str, user: str) -> str:
        import httpx

        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.post(f"{self._host}/api/chat", json=payload)
            r.raise_for_status()
            return str(r.json()["message"]["content"])


class ClaudeBackend(LLMBackend):
    def __init__(self, api_key: str, model: str) -> None:
        self._api_key = api_key
        self._model = model

    @property
    def name(self) -> str:
        return "claude"

    @property
    def model(self) -> str:
        return self._model

    async def chat(self, system: str, user: str) -> str:
        import anthropic

        client = anthropic.AsyncAnthropic(api_key=self._api_key)
        message = await client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        block = message.content[0]
        return str(block.text) if hasattr(block, "text") else ""


def make_backend(backend: str, model: str | None = None) -> LLMBackend:
    if backend == "claude":
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        m = model or os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")
        return ClaudeBackend(api_key=api_key, model=m)
    # default: ollama
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    m = model or os.environ.get("OLLAMA_MODEL", "llama3.2")
    return OllamaBackend(host=host, model=m)


class Commentator:
    def __init__(self, backend: LLMBackend) -> None:
        self._backend = backend

    async def explain(self, ctx: BoardContext, level: str = "intermediate") -> str:
        system = SYSTEM_PROMPTS.get(level, SYSTEM_PROMPTS["intermediate"])
        check_line = "\n【王手がかかっています】" if ctx.is_check else ""
        user = _USER_TEMPLATE.format(
            hands_text=ctx.hands_text,
            board_text=ctx.board_text,
            turn_text=ctx.turn_text,
            move_count=ctx.move_count,
            phase=ctx.phase,
            check_line=check_line,
            eval_text=ctx.eval_text,
            candidates_text=ctx.candidates_text,
        )
        return await self._backend.chat(system, user)

    @property
    def backend_name(self) -> str:
        return self._backend.name

    @property
    def model_name(self) -> str:
        return self._backend.model
