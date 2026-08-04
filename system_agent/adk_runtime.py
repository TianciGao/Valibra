"""
系统智能体服务的轻量级 ADK 运行时封装。

该模块在导入阶段将 ADK 依赖保持为可选状态，因此即使未安装 google-adk，
服务仍然能够启动，并报告清晰的健康状态信号。
"""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from shared.audit import normalize_usage, to_jsonable

logger = logging.getLogger(__name__)


@dataclass
class _SessionRef: # 定义一个数据类 _SessionRef，用于存储 ADK 会话的引用信息，包括应用名称、用户 ID 和会话 ID
    app_name: str
    user_id: str
    session_id: str


class AdkRuntime: 
    """Manage ADK runners and sessions for c-interact and a-interact."""

    def __init__(self) -> None: # 初始化 AdkRuntime 类的实例，设置可用性、错误信息、后端、运行器和会话引用等属性
        self.available = False
        self.error = ""
        self._backend: Optional[Dict[str, Any]] = None
        self._runners: Dict[str, Any] = {}
        self._session_refs: Dict[tuple[str, str], _SessionRef] = {}
        self._lock = asyncio.Lock()
        self._load_backend()

    def _load_backend(self) -> None: # 尝试加载 ADK 运行时的后端模块，包括 runners、genai.types 和 agent 模块，并根据可用的类设置运行器类型和会话服务类型。如果加载失败，则记录错误信息并将可用性设置为 False
        if self._backend is not None or self.error: # 如果后端已经加载或存在错误信息，则直接返回
            return
        try: # 尝试导入 ADK 相关模块，并根据可用的类设置运行器类型和会话服务类型
            runners_mod = importlib.import_module("google.adk.runners") 
            genai_types = importlib.import_module("google.genai.types")
            agent_mod = importlib.import_module("system_agent.agent")
            build_agent = getattr(agent_mod, "build_agent") # 获取 build_agent 函数，用于构建 ADK 智能体实例

            backend: Dict[str, Any] = { # 定义一个字典 backend，用于存储 ADK 运行时的后端信息，包括 types 和 build_agent 函数
                "types": genai_types,
                "build_agent": build_agent,
            }

            if hasattr(runners_mod, "InMemoryRunner"): # 如果 runners 模块中存在 InMemoryRunner 类，则将其设置为运行器类型，并将 runner_kind 设置为 "in_memory"
                backend["runner_cls"] = getattr(runners_mod, "InMemoryRunner")
                backend["runner_kind"] = "in_memory"
            else:
                backend["runner_cls"] = getattr(runners_mod, "Runner")
                sessions_mod = importlib.import_module("google.adk.sessions")
                backend["session_service_cls"] = getattr(sessions_mod, "InMemorySessionService")
                backend["runner_kind"] = "legacy"

            self._backend = backend
            self.available = True
        except Exception as exc: # 如果导入或设置后端失败，则记录错误信息并将可用性设置为 False
            self.error = str(exc)
            self.available = False
            logger.warning("ADK runtime unavailable: %s", exc)

    async def _get_runner(self, mode: str) -> tuple[Any, str]: # 定义一个异步函数 _get_runner，用于获取指定模式的 ADK 运行器实例和应用名称。如果运行器已经存在于 _runners 字典中，则直接返回；否则，根据可用的后端信息创建新的运行器实例，并将其存储在 _runners 字典中。
        if mode in self._runners: # 如果指定模式的运行器已经存在于 _runners 字典中，则直接返回该运行器实例和应用名称
            return self._runners[mode]
        if not self.available or self._backend is None: # 如果 ADK 运行时不可用或后端信息为空，则抛出运行时错误，提示 ADK 运行时不可用
            raise RuntimeError(self.error or "ADK runtime unavailable")

        app_name = f"bird_interact_{mode.replace('-', '_')}" # 根据指定模式生成应用名称，将模式中的连字符替换为下划线
        agent = self._backend["build_agent"](mode)

        if self._backend["runner_kind"] == "in_memory": # 如果后端的运行器类型为 "in_memory"，则直接创建 InMemoryRunner 实例，并传入构建的智能体实例和应用名称；否则，创建 SessionService 实例，并将其传入 Runner 实例中。
            runner = self._backend["runner_cls"](agent=agent, app_name=app_name)
        else:
            session_service = self._backend["session_service_cls"]()
            runner = self._backend["runner_cls"](
                agent=agent,
                app_name=app_name,
                session_service=session_service,
            )

        self._runners[mode] = (runner, app_name) # 将创建的运行器实例和应用名称存储在 _runners 字典中，以便后续调用时可以直接获取 
        return runner, app_name 

    def _make_text_message(self, text: str) -> Any: # 包装输入消息
        if not self.available or self._backend is None: # 检查 ADK
            raise RuntimeError(self.error or "ADK runtime unavailable")

        types_mod = self._backend["types"] # 获取后端的 types 模块，用于创建 ADK 消息对象
        try:
            return types_mod.Content(
                role="user",
                parts=[types_mod.Part(text=text)],
            )
        except TypeError: 
            return types_mod.Content(
                role="user",
                parts=[types_mod.Part.from_text(text=text)],
            )

    @staticmethod # 定义一个静态方法 _extract_text_from_content，用于从 ADK 消息内容对象中提取文本信息。如果内容对象为 None，则返回空字符串；否则，遍历内容对象的 parts 属性，获取每个 part 的 text 属性，并将其拼接为一个字符串返回。
    def _extract_text_from_content(content: Any) -> str: # 提取文字
        if content is None:
            return ""
        parts = getattr(content, "parts", None) or []
        texts = []
        for part in parts:
            text = getattr(part, "text", None)
            if text:
                texts.append(text)
        return "\n".join(texts).strip()

    @staticmethod 
    def _event_is_final(event: Any) -> bool: # 判断最终回答
        is_final = getattr(event, "is_final_response", None)
        if callable(is_final):
            try:
                return bool(is_final())
            except Exception:
                return False
        return False

    @staticmethod
    def _session_id(session: Any) -> str:
        return getattr(session, "id", None) or getattr(session, "session_id", "")

    @staticmethod
    def _preview(value: Any, limit: int = 4000) -> str: # 安全转换并限制长度
        try:
            if isinstance(value, (dict, list)):
                text = json.dumps(value, ensure_ascii=False)
            else:
                text = str(value)
        except Exception:
            text = repr(value)
        if len(text) > limit:
            return text[:limit] + "...<truncated>"
        return text

    def _serialize_part(self, part: Any) -> Dict[str, Any]: # 转换单个事件片段
        text = getattr(part, "text", None)
        if text:
            return {
                "type": "text",
                "text": text,
                "thought": bool(getattr(part, "thought", False)),
                "thought_signature": getattr(part, "thought_signature", None),
            }

        function_call = getattr(part, "function_call", None)
        if function_call is not None:
            return {
                "type": "function_call",
                "name": getattr(function_call, "name", ""),
                "id": getattr(function_call, "id", ""),
                "args": to_jsonable(getattr(function_call, "args", {}) or {}),
            }

        function_response = getattr(part, "function_response", None)
        if function_response is not None:
            return {
                "type": "function_response",
                "name": getattr(function_response, "name", ""),
                "id": getattr(function_response, "id", ""),
                "response": to_jsonable(getattr(function_response, "response", "")),
            }

        return {"type": "unknown", "repr": self._preview(part)}

    def _serialize_event(self, event: Any) -> Dict[str, Any]: # 这里开始进入生命周期了
        content = getattr(event, "content", None)
        parts = getattr(content, "parts", None) or []
        result = {
            "type": "adk_event",
            "id": getattr(event, "id", ""),
            "timestamp": str(getattr(event, "timestamp", "")),
            "author": getattr(event, "author", ""),
            "invocation_id": getattr(event, "invocation_id", ""),
            "branch": getattr(event, "branch", ""),
            "final": self._event_is_final(event),
            "partial": getattr(event, "partial", None),
            "content": {
                "role": getattr(content, "role", "") if content else "",
                "parts": [self._serialize_part(part) for part in parts],
            },
        }
        usage = getattr(event, "usage_metadata", None)
        if usage is not None:
            result["usage"] = normalize_usage(usage)
        actions = getattr(event, "actions", None)
        if actions is not None:
            result["actions"] = to_jsonable(actions)
        error_code = getattr(event, "error_code", None)
        if error_code:
            result["error_code"] = error_code
            result["error_message"] = getattr(event, "error_message", "")
        return result

    async def init_session(
        self,
        task_id: str,
        mode: str,
        state: Optional[Dict[str, Any]] = None,
        reset: bool = False,
    ) -> Dict[str, Any]:
        async with self._lock:
            runner, app_name = await self._get_runner(mode)
            key = (mode, task_id)
            if key in self._session_refs and not reset:
                ref = self._session_refs[key]
                return {
                    "task_id": task_id,
                    "mode": mode,
                    "session_id": ref.session_id,
                    "adk_available": True,
                }

            user_id = f"user_{task_id}"
            session = await runner.session_service.create_session(  # 创建一个新的 ADK 会话，并传入应用名称、用户 ID 和初始状态等信息。如果会话创建成功，则将其存储在 _session_refs 字典中，并返回会话的相关信息；否则，抛出运行时错误，提示会话创建失败。
                app_name=app_name,
                user_id=user_id,
                state=state or {},
            )
            session_state = getattr(session, "state", {}) or {}
            session_state.setdefault("tool_trajectory", [])
            session_state.setdefault("adk_events", [])
            session.state = session_state
            ref = _SessionRef(  # 创建一个 _SessionRef 实例，用于存储新创建的 ADK 会话的引用信息，包括应用名称、用户 ID 和会话 ID 
                app_name=app_name,
                user_id=user_id,
                session_id=self._session_id(session),
            )
            self._session_refs[key] = ref
            return {
                "task_id": task_id,
                "mode": mode,
                "session_id": ref.session_id,
                "adk_available": True,
            }

    async def run_turn(self, task_id: str, mode: str, message: str, # 运行 Agent 一轮
                       **kwargs) -> Dict[str, Any]:
        if not self.available:
            raise RuntimeError(self.error or "ADK runtime unavailable")

        key = (mode, task_id) # 查找任务 Session
        if key not in self._session_refs: # 检查ADK是否正常
            await self.init_session(task_id=task_id, mode=mode, state={}, reset=False)

        runner, _ = await self._get_runner(mode)
        ref = self._session_refs[key]
        new_message = self._make_text_message(message)

        # Reset per-phase flags via append_event (the ADK-native way)
        if mode == "c-interact":
            from google.adk.events import Event, EventActions
            session = await runner.session_service.get_session(
                app_name=ref.app_name, user_id=ref.user_id, session_id=ref.session_id,
            )
            if session:
                reset_event = Event(
                    author="system",
                    actions=EventActions(state_delta={"_submitted_this_phase": False}),
                )
                await runner.session_service.append_event(session, reset_event)

        final_text = ""
        turn_events = [{
            "type": "user_message",
            "message": message,
        }]
        async for event in runner.run_async( # 真正调用 Agent
            user_id=ref.user_id,
            session_id=ref.session_id,
            new_message=new_message,
        ):
            turn_events.append(self._serialize_event(event))
            content = getattr(event, "content", None)
            text = self._extract_text_from_content(content)
            if self._event_is_final(event) and text:
                final_text = text
            elif text:
                final_text = text

        session = await runner.session_service.get_session(
            app_name=ref.app_name,
            user_id=ref.user_id,
            session_id=ref.session_id,
        )
        session_state = getattr(session, "state", {}) or {}
        adk_events = session_state.get("adk_events", [])
        adk_events.extend(turn_events)
        session_state["adk_events"] = adk_events
        session.state = session_state
        return {
            "task_id": task_id,
            "mode": mode,
            "session_id": ref.session_id,
            "response": final_text,
            "state": session_state,
            "adk_available": True,
        }

    async def cleanup_session(self, task_id: str, mode: str) -> Dict[str, Any]:
        """Release one completed in-memory ADK session without affecting its saved audit."""
        async with self._lock:
            key = (mode, task_id)
            ref = self._session_refs.pop(key, None)
            if ref is None:
                return {
                    "status": "ok",
                    "task_id": task_id,
                    "mode": mode,
                    "session_removed": False,
                }
            runner, _ = await self._get_runner(mode)
            await runner.session_service.delete_session(
                app_name=ref.app_name,
                user_id=ref.user_id,
                session_id=ref.session_id,
            )
            return {
                "status": "ok",
                "task_id": task_id,
                "mode": mode,
                "session_removed": True,
            }
