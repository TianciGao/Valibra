"""Requirement Grounding 的 Updater 与 LLM 固定表单入口。

本模块提供三种 Updater：空操作、确定性规则，以及严格受约束的 LLM
版本。理解 LLM 路径时，可以按下面的数据流阅读：

``Observation``
    -> ``LLMUpdater.propose``
    -> ``LLMFrameProposal``（模型只能填写的固定表单）
    -> ``_parse_llm_frame_response``（严格 JSON、Pydantic 和原文锚定）
    -> ``_llm_patch_from_proposal``（转换为正式 Slot 与原子 Patch）
    -> ``reducer.apply_patch``（唯一的业务状态写入入口）。

其中，``LLMFrameProposal`` 只是 Provider 响应的入口合同，不是最终保存
在 Session State 中的 Requirement Frame。正式的 ``ValueSlot``、
``SchemaSlot``、``OperationSlot``、``RequirementFrame`` 和 Runtime 定义在
``requirement_grounding/models.py``。初始 ``user_query`` 生成 Slot；
``user_answer`` 和 Phase 2 follow-up 则由 ``_llm_reconciled_patch`` 按
``slot_kind + slot_role`` 与现有活动 Slot 做确定性匹配和更新。Updater
只提出 ``RequirementGroundingPatch``，绝不直接修改 Runtime；最终是否
落盘、revision 是否递增以及原子校验均由 ``requirement_grounding/reducer.py``
负责。

Frame 进入真实 Agent 生命周期后的相关入口位于：

* ``requirement_grounding/prompt_view.py`` 的公开渲染入口：把当前活动 Slot
  渲染成有界的 Agent-facing Requirement View；
* ``valibra_agent/grounding_callbacks.py::before_model_callback``：消费本轮
  用户消息，生成/更新 Frame，并按模式渲染或注入 View；
* ``valibra_agent/grounding_callbacks.py::after_tool_callback``：把工具结果
  和合法的 ``ask_user`` 回答转换成 Observation；
* ``valibra_agent/grounding_callbacks.py::_process_shadow_observation``：根据
  Observation 类型选择 LLM、Rule、NoOp 或 Evidence Bridge。

这些职责边界用于保证：模型输出必须先通过固定表单，本模块不能绕过
Reducer 写状态，Grounding 失败也不能改变 Baseline Agent 的工具、预算或
响应协议。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import uuid
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Protocol
from urllib.parse import urlsplit

from pydantic import (
    ConfigDict,
    Field,
    JsonValue,
    ValidationError,
    field_validator,
    model_validator,
)

from shared.model_presets import (
    canonical_json as preset_canonical_json,
    load_model_preset,
)
from shared.audit import to_jsonable

from valibra_agent.requirement_grounding.linguistic_hints import (
    LinguisticHint,
    extract_linguistic_hints,
)
from valibra_agent.requirement_grounding.models import (
    FAILED_FRAME_INITIALIZATION_REASONS,
    FrameInitializationReason,
    GroundingEvidence,
    GroundingSlot,
    KernelModel,
    Observation,
    OperationSlot,
    RequirementGroundingPatch,
    RequirementGroundingState,
    SchemaSlot,
    ValueSlot,
)
from valibra_agent.requirement_grounding.telemetry import (
    LLMCallTelemetryRecorder,
)


GROUNDING_LLM_ENV_NAMES = (
    "GROUNDING_UPDATER_MODE",
    "GROUNDING_MODEL_PRESET",
    "GROUNDING_TIMEOUT_SECONDS",
    "GROUNDING_MAX_TOKENS",
    "GROUNDING_MAX_CALLS_PER_TASK",
    "GROUNDING_PROMPT_SHA256",
)
GROUNDING_PROVIDER_ENV_NAMES = (
    "GROUNDING_API_BASE",
    "GROUNDING_API_KEY",
    "GROUNDING_API_KEY_FILE",
    "GROUNDING_USE_BEARER_FOR_CUSTOM_BASE",
)
GROUNDING_LLM_OBSERVATION_TYPES = frozenset({"user_query", "user_answer"})

_BEARER_TOKEN_RE = re.compile(
    r"Authorization:\s*Bearer\s+([^\"'\\\s]+)",
    flags=re.IGNORECASE,
)
_SENSITIVE_AUDIT_KEYS = frozenset(
    {
        "api_key",
        "authorization",
        "password",
        "secret",
        "token",
        "access_token",
        "refresh_token",
        "extra_headers",
        "headers",
    }
)

# 这是发给 Grounding 模型的冻结 Prompt；其 SHA 用于保证实验可复现。
LLM_FRAME_PROMPT = """Fill the fixed Valibra requirement-frame form from bounded JSON input.
You are filling a form, not designing a data structure. Return exactly one JSON
object and no Markdown, prose, comments, or additional fields.

The top-level fields must be exactly: proposal_outcome, value_slots,
schema_slots, operation_slots, ambiguities. proposal_outcome must be exactly one
of populated, insufficient_information, or no_extractable_requirement.
ambiguities must always be an empty array.

Use populated if and only if at least one value, schema, or operation slot is
present. Use insufficient_information only when observation_text expresses a
requirement but the bounded supplied input does not contain enough information
to fill even one safe slot. Use no_extractable_requirement only when
observation_text contains no requirement expressible as a value, schema, or
operation slot. Both empty outcomes require all three slot arrays to be empty;
never use an empty outcome when any slot is present.

Each value slot must contain exactly: slot_role, mention, interpretation,
value_type. Each schema slot must contain exactly: slot_role, mention,
interpretation. Schema slots are natural-language candidates only: never emit
a real table, column, database identifier, binding, confidence, or hidden fact.
Each operation slot must contain exactly: slot_role, mention, interpretation,
operation_type, parameters. operation_type must be exactly one of projection,
filter, aggregation, group, order, limit, distinct, or other. If no listed type
matches exactly, use other; never invent a new value. parameters MUST always be
a JSON object. Values inside that object may be JSON scalars only, never nested
objects or arrays.

Correct: "parameters": {"direction": "desc"}
Correct: "parameters": {"limit": 5}
Correct when there are no parameters: "parameters": {}
Wrong: "parameters": "desc"
Wrong: "parameters": 5
Wrong: "parameters": true

Every mention must be non-empty and copied verbatim as one continuous,
case-sensitive substring of observation_text. Put any paraphrase or explanation
only in interpretation. Never rewrite a mention.

Only use observation_text, the supplied Observation, and current bounded State.
All output is provisional/hypothesized. Never infer from ground truth, test
cases, hidden follow-up, schema contents, prompt history, audit logs, tools,
network resources, or outside knowledge."""
LLM_FRAME_PROMPT_SHA256 = hashlib.sha256(
    LLM_FRAME_PROMPT.encode("utf-8")
).hexdigest()

MAX_LLM_FRAME_INPUT_CHARS = 262_144
MAX_LLM_FRAME_RESPONSE_CHARS = 65_536
MAX_LLM_FRAME_SLOTS = 64
LLMFrameProposalOutcome = Literal[
    "populated",
    "insufficient_information",
    "no_extractable_requirement",
]


class _StrictLLMModel(KernelModel):
    """LLM 边界模型使用严格类型，不接受自动类型转换。"""

    model_config = ConfigDict(
        **{
            **KernelModel.model_config,
            "strict": True,
            "str_strip_whitespace": False,
        }
    )


class GroundingLLMConfig(_StrictLLMModel):
    """一次 LLM Updater 实验的冻结、自校验配置。"""

    mode: Literal["llm"] = "llm"
    model_preset: str = Field(min_length=1, max_length=128)
    preset_config_json: str = Field(min_length=2, max_length=8192)
    preset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    timeout_seconds: float = Field(gt=0.0, le=600.0)
    max_tokens: int = Field(ge=1, le=131_072)
    max_calls_per_task: int = Field(ge=1, le=128)
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    form_schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    configuration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_frozen_hashes(self) -> "GroundingLLMConfig":
        if self.prompt_sha256 != LLM_FRAME_PROMPT_SHA256:
            raise ValueError("GROUNDING_PROMPT_SHA256 does not match frozen prompt")
        if self.form_schema_sha256 != LLM_FRAME_FORM_SCHEMA_SHA256:
            raise ValueError("LLM Frame form schema SHA256 mismatch")
        try:
            preset_config = json.loads(self.preset_config_json)
        except json.JSONDecodeError as exc:
            raise ValueError("preset_config_json must be valid JSON") from exc
        if not isinstance(preset_config, dict):
            raise ValueError("preset_config_json must encode an object")
        if preset_canonical_json(preset_config) != self.preset_config_json:
            raise ValueError("preset_config_json must be canonical JSON")
        actual_preset_sha = hashlib.sha256(
            self.preset_config_json.encode("utf-8")
        ).hexdigest()
        if actual_preset_sha != self.preset_sha256:
            raise ValueError("normalized model preset SHA256 mismatch")
        preset_max_tokens = preset_config.get("max_tokens")
        if (
            isinstance(preset_max_tokens, bool)
            or not isinstance(preset_max_tokens, int)
            or self.max_tokens > preset_max_tokens
        ):
            raise ValueError(
                "GROUNDING_MAX_TOKENS must not exceed preset max_tokens"
            )
        if self.configuration_sha256 != _grounding_configuration_sha256(
            mode=self.mode,
            model_preset=self.model_preset,
            preset_sha256=self.preset_sha256,
            timeout_seconds=self.timeout_seconds,
            max_tokens=self.max_tokens,
            max_calls_per_task=self.max_calls_per_task,
            prompt_sha256=self.prompt_sha256,
            form_schema_sha256=self.form_schema_sha256,
        ):
            raise ValueError("grounding configuration SHA256 mismatch")
        return self

    @property
    def preset_config(self) -> dict[str, JsonValue]:
        """读取已校验的模型预设 JSON。"""

        return json.loads(self.preset_config_json)


class GroundingLLMRequest(_StrictLLMModel):
    """发送给可注入 LLM 客户端的完整请求。"""

    prompt: str = Field(min_length=1, max_length=MAX_LLM_FRAME_INPUT_CHARS)
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    form_schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    configuration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_preset: str = Field(min_length=1, max_length=128)
    preset_config: dict[str, JsonValue]
    max_tokens: int = Field(ge=1, le=131_072)
    timeout_seconds: float = Field(gt=0.0, le=600.0)


class GroundingLLMUsage(_StrictLLMModel):
    """单次 Grounding 模型的 Token 和费用。"""

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    reasoning_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    cost: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def validate_finite_cost(self) -> "GroundingLLMUsage":
        if self.cost is not None and not math.isfinite(self.cost):
            raise ValueError("LLM cost must be finite")
        return self


class GroundingLLMResponse(_StrictLLMModel):
    """LLM 客户端返回的正文、用量和审计摘要。"""

    content: str = Field(max_length=MAX_LLM_FRAME_RESPONSE_CHARS)
    usage: GroundingLLMUsage = Field(default_factory=GroundingLLMUsage)
    model: str = Field(default="", max_length=256)
    provider: str = Field(default="", max_length=128)
    credential_source: Literal["", "direct", "file"] = ""
    request_sha256: str = Field(default="", pattern=r"^(?:|[0-9a-f]{64})$")
    response_sha256: str = Field(default="", pattern=r"^(?:|[0-9a-f]{64})$")
    raw_audit_ref: str = Field(default="", max_length=1024)


class GroundingProviderConfig(_StrictLLMModel):
    """不含密钥正文、可自校验的 Provider 连接配置。"""

    api_base: str = Field(min_length=1, max_length=2048)
    model_id: str = Field(min_length=1, max_length=256)
    llm_configuration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    credential_source: Literal["direct", "file"]
    api_key_file: str | None = Field(default=None, max_length=2048)
    use_bearer_for_custom_base: bool = False
    retry_count: Literal[0] = 0
    raw_audit_dir: str = Field(min_length=1, max_length=4096)
    configuration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_provider_config(self) -> "GroundingProviderConfig":
        parsed = urlsplit(self.api_base)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("GROUNDING_API_BASE must be an absolute HTTP(S) URL")
        if self.credential_source == "file" and not self.api_key_file:
            raise ValueError("file credential source requires api_key_file")
        if self.credential_source == "direct" and self.api_key_file is not None:
            raise ValueError("direct credential source cannot retain api_key_file")
        expected = _provider_configuration_sha256(
            api_base=self.api_base,
            model_id=self.model_id,
            llm_configuration_sha256=self.llm_configuration_sha256,
            credential_source=self.credential_source,
            use_bearer_for_custom_base=self.use_bearer_for_custom_base,
            retry_count=self.retry_count,
        )
        if self.configuration_sha256 != expected:
            raise ValueError("Grounding Provider configuration SHA256 mismatch")
        return self


class GroundingProviderError(RuntimeError):
    """已脱敏的 Provider 错误，不包含响应正文或密钥。"""


class LLMFrameUpdateError(ValueError):
    """LLM Frame 边界的有限失败类型；不依赖异常正文做分类。"""

    def __init__(self, reason: FrameInitializationReason) -> None:
        if reason not in FAILED_FRAME_INITIALIZATION_REASONS:
            raise ValueError("LLM Frame update error requires a failure reason")
        self.reason = reason
        super().__init__(reason)


class AsyncGroundingLLMClient(Protocol):
    """可替换的异步客户端边界，测试时可注入 fake client。"""

    provider_may_continue_after_cancel: bool
    provider_may_bill_after_cancel: bool

    async def complete(
        self,
        request: GroundingLLMRequest,
    ) -> GroundingLLMResponse | Mapping[str, Any]: ...


class LiteLLMGroundingClient:
    """LiteLLM 异步适配器：凭据独立，原始审计单独落盘。"""

    provider_may_continue_after_cancel = True
    provider_may_bill_after_cancel = True

    def __init__(
        self,
        config: GroundingProviderConfig,
        *,
        environment: Mapping[str, str] | None = None,
        completion: Callable[..., Awaitable[Any]] | None = None,
    ) -> None:
        """保存已校验配置，并允许测试注入 completion 函数。"""

        self.config = GroundingProviderConfig.model_validate(
            config.model_dump(mode="python")
        )
        self._environment = environment if environment is not None else os.environ
        self._completion = completion

    async def complete(self, request: GroundingLLMRequest) -> GroundingLLMResponse:
        """调用 Provider，保存脱敏审计，再返回统一响应。"""

        if request.preset_config.get("model") != self.config.model_id:
            raise GroundingProviderError("Grounding request model mismatch")
        if request.configuration_sha256 != self.config.llm_configuration_sha256:
            raise GroundingProviderError("Grounding LLM configuration mismatch")

        # 密钥只在发请求和落盘前的泄漏检查中使用，不进入配置或返回值。
        api_key = _read_grounding_api_key(self.config, self._environment)
        request_audit, provider_kwargs = _build_provider_request(
            request,
            self.config,
            api_key=api_key,
        )
        request_sha = _stable_json_sha256(request_audit)
        audit_path = _new_private_audit_path(self.config, request_sha)
        completion = self._completion
        if completion is None:
            import litellm

            completion = litellm.acompletion

        try:
            response = await completion(**provider_kwargs)
        except Exception as exc:
            # 失败审计只保存错误类型，不保存可能带敏感信息的异常正文。
            _write_private_provider_audit(
                audit_path,
                {
                    "schema_version": "1.0",
                    "status": "failed",
                    "request": request_audit,
                    "request_sha256": request_sha,
                    "error_type": type(exc).__name__[:128],
                },
                api_key=api_key,
            )
            raise GroundingProviderError(
                f"Grounding Provider request failed ({type(exc).__name__[:128]})"
            ) from None

        # 原始响应先递归移除敏感字段，再计算指纹和写入私有目录。
        raw_response = _sanitize_audit_value(to_jsonable(response))
        response_sha = _stable_json_sha256(raw_response)
        _write_private_provider_audit(
            audit_path,
            {
                "schema_version": "1.0",
                "status": "succeeded",
                "request": request_audit,
                "request_sha256": request_sha,
                "response": raw_response,
                "response_sha256": response_sha,
            },
            api_key=api_key,
        )
        content = _provider_response_content(response)
        usage = _provider_usage(response)
        provider = _provider_name(response)
        return GroundingLLMResponse(
            content=content,
            usage=usage,
            model=self.config.model_id,
            provider=provider,
            credential_source=self.config.credential_source,
            request_sha256=request_sha,
            response_sha256=response_sha,
            raw_audit_ref=_private_audit_ref(self.config, audit_path),
        )


class LLMValueSlotProposal(_StrictLLMModel):
    """模型提出的值槽位候选。"""

    slot_role: str = Field(min_length=1, max_length=64)
    mention: str = Field(min_length=1, max_length=256)
    interpretation: str = Field(min_length=1, max_length=512)
    value_type: str | None = Field(max_length=64)


class LLMSchemaSlotProposal(_StrictLLMModel):
    """模型提出的 Schema 概念；此时不允许绑定真实字段。"""

    slot_role: str = Field(min_length=1, max_length=64)
    mention: str = Field(min_length=1, max_length=256)
    interpretation: str = Field(min_length=1, max_length=512)


class LLMOperationSlotProposal(_StrictLLMModel):
    """模型提出的查询操作候选。"""

    slot_role: str = Field(min_length=1, max_length=64)
    mention: str = Field(min_length=1, max_length=256)
    interpretation: str = Field(min_length=1, max_length=512)
    operation_type: Literal[
        "projection",
        "filter",
        "aggregation",
        "group",
        "order",
        "limit",
        "distinct",
        "other",
    ]
    parameters: dict[str, str | int | float | bool | None]

    @field_validator("parameters")
    @classmethod
    def validate_scalar_parameters(
        cls,
        value: dict[str, str | int | float | bool | None],
    ) -> dict[str, str | int | float | bool | None]:
        """限制表单参数的数量、长度和有限数值，不做类型转换。"""

        if len(value) > 16:
            raise ValueError("operation parameters exceed item limit")
        for key, item in value.items():
            if len(key) > 64:
                raise ValueError("operation parameter key exceeds length limit")
            if isinstance(item, str) and len(item) > 256:
                raise ValueError("operation parameter value exceeds length limit")
            if isinstance(item, float) and not math.isfinite(item):
                raise ValueError("operation parameter must be finite")
        return value


class LLMFrameProposal(_StrictLLMModel):
    """模型输出的完整临时 Frame；当前明确禁止生成歧义。"""

    proposal_outcome: LLMFrameProposalOutcome
    value_slots: list[LLMValueSlotProposal] = Field(max_length=MAX_LLM_FRAME_SLOTS)
    schema_slots: list[LLMSchemaSlotProposal] = Field(max_length=MAX_LLM_FRAME_SLOTS)
    operation_slots: list[LLMOperationSlotProposal] = Field(max_length=MAX_LLM_FRAME_SLOTS)
    ambiguities: list[dict[str, JsonValue]] = Field(max_length=0)

    @model_validator(mode="after")
    def validate_outcome_and_total_slot_limit(self) -> "LLMFrameProposal":
        total = len(self.value_slots) + len(self.schema_slots) + len(self.operation_slots)
        if total > MAX_LLM_FRAME_SLOTS:
            raise ValueError("LLM Frame proposal exceeds total slot limit")
        if self.proposal_outcome == "populated" and total == 0:
            raise ValueError("populated LLM Frame proposal requires at least one Slot")
        if self.proposal_outcome != "populated" and total != 0:
            raise ValueError("empty LLM Frame proposal outcome requires zero Slots")
        return self


# 机器可读表单只从上面的严格 Pydantic 模型生成，Provider 与本地校验共用。
LLM_FRAME_FORM_SCHEMA = LLMFrameProposal.model_json_schema()
LLM_FRAME_FORM_SCHEMA_JSON = preset_canonical_json(LLM_FRAME_FORM_SCHEMA)
LLM_FRAME_FORM_SCHEMA_SHA256 = hashlib.sha256(
    LLM_FRAME_FORM_SCHEMA_JSON.encode("utf-8")
).hexdigest()


class LLMUpdaterResult(_StrictLLMModel):
    """一次有效模型提案及其原子 Patch；outcome 不写入业务状态。"""

    proposal_outcome: LLMFrameProposalOutcome
    patch: RequirementGroundingPatch


class LLMUpdater:
    """异步 LLM Frame Updater；当前尚未接入在线 Agent 回调。"""

    mode = "llm"

    def __init__(
        self,
        client: AsyncGroundingLLMClient,
        config: GroundingLLMConfig,
    ) -> None:
        """绑定客户端，并复制一份已校验的冻结配置。"""

        self.client = client
        self.config = GroundingLLMConfig.model_validate(
            config.model_dump(mode="python")
        )

    async def propose(
        self,
        observation: Observation,
        state: RequirementGroundingState,
        *,
        base_revision: int,
        telemetry: LLMCallTelemetryRecorder,
    ) -> RequirementGroundingPatch:
        """兼容入口：只返回 Patch，保留 P4.2a 的公开合同。"""

        result = await self.propose_result(
            observation,
            state,
            base_revision=base_revision,
            telemetry=telemetry,
        )
        return result.patch

    async def propose_result(
        self,
        observation: Observation,
        state: RequirementGroundingState,
        *,
        base_revision: int,
        telemetry: LLMCallTelemetryRecorder,
    ) -> LLMUpdaterResult:
        """把模型的有限 outcome 与 Patch 作为瞬态类型化结果返回。"""

        if observation.observation_type not in GROUNDING_LLM_OBSERVATION_TYPES:
            raise ValueError(
                "LLMUpdater accepts only user_query and user_answer observations"
            )
        detached_observation = Observation.model_validate(
            observation.model_dump(mode="python")
        )
        detached_state = RequirementGroundingState.model_validate(
            state.model_dump(mode="python")
        )
        request = _build_llm_request(
            detached_observation,
            detached_state,
            base_revision=base_revision,
            config=self.config,
        )
        # 从这里起请求可能已产生费用，因此先标记 attempted。
        telemetry.mark_provider_attempted()
        raw_response = await self.client.complete(request)
        try:
            response = GroundingLLMResponse.model_validate(raw_response)
        except ValidationError as exc:
            raise LLMFrameUpdateError("transport_format_invalid") from exc
        telemetry.capture_usage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            reasoning_tokens=response.usage.reasoning_tokens,
            total_tokens=(
                response.usage.total_tokens
                or response.usage.input_tokens + response.usage.output_tokens
            ),
            cost=response.usage.cost,
        )
        telemetry.capture_provider_audit(
            model=response.model,
            provider=response.provider,
            credential_source=response.credential_source,
            request_sha256=response.request_sha256,
            response_sha256=response.response_sha256,
            raw_audit_ref=response.raw_audit_ref,
        )
        try:
            proposal = _parse_llm_frame_response(
                response.content,
                observation_text=_observation_text(detached_observation),
            )
        except LLMFrameUpdateError:
            raise
        except ValidationError as exc:
            raise LLMFrameUpdateError("form_validation_failed") from exc
        try:
            patch = _llm_patch_from_proposal(
                detached_observation,
                detached_state,
                proposal,
                base_revision=base_revision,
            )
            return LLMUpdaterResult(
                proposal_outcome=proposal.proposal_outcome,
                patch=patch,
            )
        except LLMFrameUpdateError:
            raise
        except Exception as exc:
            raise LLMFrameUpdateError("patch_rejected") from exc


def load_grounding_llm_config(
    project_root: Path,
    environment: Mapping[str, str] | None = None,
) -> GroundingLLMConfig:
    """读取但不激活模型预设，并冻结全部 LLM 实验参数。"""

    source = environment if environment is not None else os.environ
    missing = [name for name in GROUNDING_LLM_ENV_NAMES if not source.get(name)]
    if missing:
        raise ValueError(f"missing frozen Grounding configuration: {missing}")
    if source["GROUNDING_UPDATER_MODE"] != "llm":
        raise ValueError("GROUNDING_UPDATER_MODE must be exactly 'llm'")
    if source["GROUNDING_PROMPT_SHA256"] != LLM_FRAME_PROMPT_SHA256:
        raise ValueError("GROUNDING_PROMPT_SHA256 does not match frozen prompt")

    preset = load_model_preset(project_root, source["GROUNDING_MODEL_PRESET"])
    timeout_seconds = _parse_positive_float(
        source["GROUNDING_TIMEOUT_SECONDS"],
        "GROUNDING_TIMEOUT_SECONDS",
    )
    max_tokens = _parse_positive_int(
        source["GROUNDING_MAX_TOKENS"],
        "GROUNDING_MAX_TOKENS",
    )
    max_calls = _parse_positive_int(
        source["GROUNDING_MAX_CALLS_PER_TASK"],
        "GROUNDING_MAX_CALLS_PER_TASK",
    )
    preset_json = preset_canonical_json(preset.normalized_config)
    config_sha = _grounding_configuration_sha256(
        mode="llm",
        model_preset=preset.name,
        preset_sha256=preset.normalized_sha256,
        timeout_seconds=timeout_seconds,
        max_tokens=max_tokens,
        max_calls_per_task=max_calls,
        prompt_sha256=LLM_FRAME_PROMPT_SHA256,
        form_schema_sha256=LLM_FRAME_FORM_SCHEMA_SHA256,
    )
    return GroundingLLMConfig(
        mode="llm",
        model_preset=preset.name,
        preset_config_json=preset_json,
        preset_sha256=preset.normalized_sha256,
        timeout_seconds=timeout_seconds,
        max_tokens=max_tokens,
        max_calls_per_task=max_calls,
        prompt_sha256=LLM_FRAME_PROMPT_SHA256,
        form_schema_sha256=LLM_FRAME_FORM_SCHEMA_SHA256,
        configuration_sha256=config_sha,
    )


def load_grounding_provider_config(
    project_root: Path,
    llm_config: GroundingLLMConfig,
    environment: Mapping[str, str] | None = None,
) -> GroundingProviderConfig:
    """只读取显式 GROUNDING_* 连接配置，不借用主模型或用户模型配置。"""

    source = environment if environment is not None else os.environ
    api_base = str(source.get("GROUNDING_API_BASE", "")).strip()
    direct_key_present = bool(str(source.get("GROUNDING_API_KEY", "")).strip())
    key_file_value = str(source.get("GROUNDING_API_KEY_FILE", "")).strip()
    if not api_base:
        raise ValueError("missing Grounding Provider variable: GROUNDING_API_BASE")
    # 直传密钥和密钥文件必须二选一，不能同时配置或同时为空。
    if direct_key_present == bool(key_file_value):
        raise ValueError(
            "configure exactly one of GROUNDING_API_KEY or GROUNDING_API_KEY_FILE"
        )

    credential_source: Literal["direct", "file"]
    resolved_key_file: str | None = None
    if key_file_value:
        credential_source = "file"
        key_path = Path(key_file_value).expanduser()
        if not key_path.is_absolute():
            key_path = project_root / key_path
        key_path = key_path.resolve()
        try:
            if not key_path.is_file() or key_path.stat().st_size <= 0:
                raise ValueError("GROUNDING_API_KEY_FILE must be a non-empty file")
            with key_path.open("rb") as handle:
                if not handle.readable():
                    raise ValueError("GROUNDING_API_KEY_FILE must be readable")
        except OSError as exc:
            raise ValueError("GROUNDING_API_KEY_FILE is not readable") from exc
        resolved_key_file = str(key_path)
    else:
        credential_source = "direct"

    use_bearer = _parse_bool(
        str(source.get("GROUNDING_USE_BEARER_FOR_CUSTOM_BASE", "false")),
        "GROUNDING_USE_BEARER_FOR_CUSTOM_BASE",
    )
    model_id = str(llm_config.preset_config.get("model", ""))
    if not model_id:
        raise ValueError("Grounding model preset does not contain a model ID")
    provider_sha = _provider_configuration_sha256(
        api_base=api_base,
        model_id=model_id,
        llm_configuration_sha256=llm_config.configuration_sha256,
        credential_source=credential_source,
        use_bearer_for_custom_base=use_bearer,
        retry_count=0,
    )
    return GroundingProviderConfig(
        api_base=api_base,
        model_id=model_id,
        llm_configuration_sha256=llm_config.configuration_sha256,
        credential_source=credential_source,
        api_key_file=resolved_key_file,
        use_bearer_for_custom_base=use_bearer,
        retry_count=0,
        raw_audit_dir=str(
            (project_root / "research-runtime" / "grounding-llm").resolve()
        ),
        configuration_sha256=provider_sha,
    )


class NoOpUpdater:
    """生成确定性的空 Patch，用来验证链路而不改业务状态。"""

    mode = "noop"

    def propose(
        self,
        observation: Observation,
        state: RequirementGroundingState,
        *,
        base_revision: int,
    ) -> RequirementGroundingPatch:
        """消费 Observation，但不提出任何业务变更。"""

        # 仍复制并校验 State，用来尽早发现调用方传入的坏数据。
        RequirementGroundingState.model_validate(state.model_dump(mode="python"))
        return RequirementGroundingPatch(
            patch_id=f"noop:{observation.observation_id}",
            base_revision=base_revision,
            source_observation_ids=(observation.observation_id,),
            diagnostics=("K0 NoOp updater: no business-state operations",),
        )


class RuleUpdater:
    """用有限的确定性语言规则生成临时 Frame Patch。"""

    mode = "rule"

    def propose(
        self,
        observation: Observation,
        state: RequirementGroundingState,
        *,
        base_revision: int,
    ) -> RequirementGroundingPatch:
        """只从用户问题或回答中提取高精度规则线索。"""

        detached_state = RequirementGroundingState.model_validate(
            state.model_dump(mode="python")
        )
        if observation.observation_type not in {"user_query", "user_answer"}:
            return _rule_noop_patch(observation, base_revision)

        hints = extract_linguistic_hints(_observation_text(observation))
        if not hints:
            return _rule_noop_patch(observation, base_revision)

        evidence_id = f"evidence.{observation.observation_id}"
        evidence = GroundingEvidence(
            evidence_id=evidence_id,
            observation_id=observation.observation_id,
            source_type=f"rule_{observation.observation_type}",
            phase=observation.phase,
            summary=observation.summary,
            raw_digest=observation.raw_digest,
            raw_log_ref=observation.raw_log_ref,
            sequence=observation.sequence,
            timestamp=None,
        )

        existing = _slot_index(detached_state)
        additions = []
        updates = []
        # 用户回答更新旧槽位时必须唯一匹配，匹配不清就宁可跳过。
        answer_role_counts = Counter(
            (hint.slot_kind, hint.slot_role) for hint in hints
        )
        for hint in hints:
            stable_id = _slot_id(hint)
            current = existing.get(stable_id)
            if current is not None and current.lifecycle != "active":
                continue
            if current is None and observation.observation_type == "user_answer":
                if answer_role_counts[(hint.slot_kind, hint.slot_role)] != 1:
                    continue
                candidates = [
                    slot
                    for slot in existing.values()
                    if slot.lifecycle == "active"
                    and slot.slot_kind == hint.slot_kind
                    and slot.slot_role == hint.slot_role
                ]
                if len(candidates) > 1:
                    continue
                if len(candidates) == 1:
                    current = candidates[0]

            slot = _slot_from_hint(
                hint,
                slot_id=current.slot_id if current is not None else stable_id,
                evidence_id=evidence_id,
                observation=observation,
                current=current,
            )
            if current is None:
                additions.append(slot)
                existing[slot.slot_id] = slot
            else:
                updates.append(slot)
                existing[slot.slot_id] = slot

        if not additions and not updates:
            return _rule_noop_patch(observation, base_revision)
        return RequirementGroundingPatch(
            patch_id=f"rule.{observation.observation_id}",
            base_revision=base_revision,
            source_observation_ids=(observation.observation_id,),
            slot_additions=tuple(additions),
            slot_updates=tuple(updates),
            ambiguity_additions=(),
            ambiguity_updates=(),
            evidence_additions=(evidence,),
            diagnostics=(
                "P4.1 deterministic provisional Frame; no ambiguity or schema binding",
            ),
        )


def build_noop_patch(
    observation: Observation,
    state: RequirementGroundingState,
    *,
    base_revision: int,
) -> RequirementGroundingPatch:
    """函数式入口：构建一个 NoOp Patch。"""

    return NoOpUpdater().propose(
        observation,
        state,
        base_revision=base_revision,
    )


def build_rule_patch(
    observation: Observation,
    state: RequirementGroundingState,
    *,
    base_revision: int,
) -> RequirementGroundingPatch:
    """函数式入口：构建一个规则 Patch。"""

    return RuleUpdater().propose(
        observation,
        state,
        base_revision=base_revision,
    )


def _rule_noop_patch(
    observation: Observation,
    base_revision: int,
) -> RequirementGroundingPatch:
    """规则没有足够把握时返回可审计的空 Patch。"""

    return RequirementGroundingPatch(
        patch_id=f"rule-noop.{observation.observation_id}",
        base_revision=base_revision,
        source_observation_ids=(observation.observation_id,),
        diagnostics=("P4.1 RuleUpdater found no safe provisional Frame change",),
    )


def _observation_text(observation: Observation) -> str:
    """取规则要分析的文本，不读取历史、隐藏状态或审计日志。"""

    # P3 会把工具字符串规范成 JSON；这里只解码这一种明确格式。
    text = observation.summary
    if observation.observation_type == "user_answer" and text.startswith('"'):
        try:
            decoded = json.loads(text)
        except (TypeError, ValueError):
            return text
        if isinstance(decoded, str):
            return decoded
    return text


def _slot_index(state: RequirementGroundingState) -> dict[str, GroundingSlot]:
    """把三类槽位合并成按 slot_id 查询的索引。"""

    frame = state.requirement_frame
    return {
        slot.slot_id: slot
        for slot in (
            *frame.value_slots,
            *frame.schema_slots,
            *frame.operation_slots,
        )
    }


def _slot_id(hint: LinguisticHint) -> str:
    """从稳定 Hint ID 派生稳定 Slot ID。"""

    return f"slot.{hint.slot_kind}.{hint.hint_id.removeprefix('hint.')}"


def _slot_from_hint(
    hint: LinguisticHint,
    *,
    slot_id: str,
    evidence_id: str,
    observation: Observation,
    current: GroundingSlot | None,
) -> GroundingSlot:
    """把规则 Hint 新建或合并成对应类型的槽位。"""

    existing_evidence = tuple(getattr(current, "evidence_refs", ()))
    evidence_refs = tuple(dict.fromkeys((*existing_evidence, evidence_id)))
    introduced_in_phase = getattr(
        current,
        "introduced_in_phase",
        observation.phase,
    )
    common = {
        "slot_id": slot_id,
        "slot_role": hint.slot_role,
        "mention": hint.mention,
        "current_interpretation": hint.interpretation,
        "grounding_status": "hypothesized",
        "evidence_refs": evidence_refs,
        "ambiguity_refs": tuple(getattr(current, "ambiguity_refs", ())),
        "origin": "rule_provisional",
        "lifecycle": "active",
        "introduced_in_phase": introduced_in_phase,
        "last_updated_phase": observation.phase,
        "sequence": observation.sequence,
    }
    if hint.slot_kind == "value":
        return ValueSlot(**common, value_type=hint.value_type)
    if hint.slot_kind == "schema":
        return SchemaSlot(
            **common,
            binding_type="unknown",
            bound_identifier=None,
        )
    if hint.slot_kind == "operation":
        return OperationSlot(
            **common,
            operation_type=hint.operation_type or "other",
            parameters=dict(hint.parameters),
        )
    raise ValueError(f"unsupported deterministic hint kind: {hint.slot_kind}")


def _grounding_configuration_sha256(
    *,
    mode: str,
    model_preset: str,
    preset_sha256: str,
    timeout_seconds: float,
    max_tokens: int,
    max_calls_per_task: int,
    prompt_sha256: str,
    form_schema_sha256: str,
) -> str:
    """计算 LLM 实验参数的稳定指纹。"""

    payload = {
        "GROUNDING_UPDATER_MODE": mode,
        "GROUNDING_MODEL_PRESET": model_preset,
        "GROUNDING_MODEL_PRESET_SHA256": preset_sha256,
        "GROUNDING_TIMEOUT_SECONDS": timeout_seconds,
        "GROUNDING_MAX_TOKENS": max_tokens,
        "GROUNDING_MAX_CALLS_PER_TASK": max_calls_per_task,
        "GROUNDING_PROMPT_SHA256": prompt_sha256,
        "LLM_FRAME_FORM_SCHEMA_SHA256": form_schema_sha256,
    }
    return hashlib.sha256(
        preset_canonical_json(payload).encode("utf-8")
    ).hexdigest()


def _provider_configuration_sha256(
    *,
    api_base: str,
    model_id: str,
    llm_configuration_sha256: str,
    credential_source: str,
    use_bearer_for_custom_base: bool,
    retry_count: int,
) -> str:
    """计算不含密钥的 Provider 配置指纹。"""

    payload = {
        "GROUNDING_API_BASE": api_base,
        "GROUNDING_MODEL_ID": model_id,
        "GROUNDING_LLM_CONFIGURATION_SHA256": llm_configuration_sha256,
        "GROUNDING_CREDENTIAL_SOURCE": credential_source,
        "GROUNDING_USE_BEARER_FOR_CUSTOM_BASE": use_bearer_for_custom_base,
        "GROUNDING_PROVIDER_RETRY_COUNT": retry_count,
    }
    return hashlib.sha256(
        preset_canonical_json(payload).encode("utf-8")
    ).hexdigest()


def _parse_positive_int(raw: str, name: str) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _parse_positive_float(raw: str, name: str) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive number") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return value


def _parse_bool(raw: str, name: str) -> bool:
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


def _build_llm_request(
    observation: Observation,
    state: RequirementGroundingState,
    *,
    base_revision: int,
    config: GroundingLLMConfig,
) -> GroundingLLMRequest:
    """把有限 Observation 和 State 拼成冻结 Prompt 请求。"""

    payload = {
        "base_revision": base_revision,
        "observation": observation.model_dump(mode="json"),
        "observation_text": _observation_text(observation),
        "state": state.model_dump(mode="json"),
    }
    dynamic_input = preset_canonical_json(payload)
    prompt = f"{LLM_FRAME_PROMPT}\n\nINPUT_JSON\n{dynamic_input}"
    if len(prompt) > MAX_LLM_FRAME_INPUT_CHARS:
        raise ValueError("bounded LLM Frame input exceeds character limit")
    return GroundingLLMRequest(
        prompt=prompt,
        prompt_sha256=config.prompt_sha256,
        configuration_sha256=config.configuration_sha256,
        model_preset=config.model_preset,
        preset_config=config.preset_config,
        max_tokens=config.max_tokens,
        timeout_seconds=config.timeout_seconds,
        form_schema_sha256=config.form_schema_sha256,
    )


def _read_grounding_api_key(
    config: GroundingProviderConfig,
    environment: Mapping[str, str],
) -> str:
    """从显式环境变量或文件读取 Grounding 专用密钥。"""

    if config.credential_source == "direct":
        value = str(environment.get("GROUNDING_API_KEY", "")).strip()
        if not value:
            raise GroundingProviderError("Grounding direct credential is unavailable")
        return value

    path = Path(config.api_key_file or "")
    try:
        content = path.read_text(encoding="utf-8")
    except OSError:
        raise GroundingProviderError("Grounding credential file is unavailable") from None
    match = _BEARER_TOKEN_RE.search(content)
    if match:
        return match.group(1)
    nonempty_lines = [line.strip() for line in content.splitlines() if line.strip()]
    if (
        len(nonempty_lines) == 1
        and not any(character.isspace() for character in nonempty_lines[0])
    ):
        return nonempty_lines[0]
    raise GroundingProviderError("Grounding credential file format is invalid")


def _build_provider_request(
    request: GroundingLLMRequest,
    config: GroundingProviderConfig,
    *,
    api_key: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """同时构造可发送参数和不含密钥的审计参数。"""

    if request.form_schema_sha256 != LLM_FRAME_FORM_SCHEMA_SHA256:
        raise GroundingProviderError("Grounding form schema mismatch")
    preset = request.preset_config
    messages = [{"role": "user", "content": request.prompt}]
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "valibra_requirement_frame",
            "strict": True,
            "schema": json.loads(LLM_FRAME_FORM_SCHEMA_JSON),
        },
    }
    generation: dict[str, Any] = {
        "temperature": preset.get("temperature", 0.0),
        "max_tokens": request.max_tokens,
    }
    if "top_p" in preset:
        generation["top_p"] = preset["top_p"]
    extra_body: dict[str, Any] = {}
    if "thinking" in preset:
        extra_body["thinking"] = preset["thinking"]
    if "reasoning_effort" in preset:
        extra_body["reasoning_effort"] = preset["reasoning_effort"]

    provider_kwargs: dict[str, Any] = {
        "model": config.model_id,
        "messages": messages,
        **generation,
        "stream": False,
        "timeout": request.timeout_seconds,
        "num_retries": 0,
        "max_retries": 0,
        "api_base": config.api_base,
        "api_key": api_key,
        "response_format": response_format,
    }
    if extra_body:
        provider_kwargs["extra_body"] = extra_body
    if config.use_bearer_for_custom_base:
        provider_kwargs["use_bearer_for_custom_base"] = True

    request_audit = {
        "model": config.model_id,
        "messages": messages,
        "generation": generation,
        "extra_body": extra_body,
        "timeout_seconds": request.timeout_seconds,
        "provider_retry_count": 0,
        "tools": [],
        "tool_choice": None,
        "response_format": response_format,
        "connection": {
            "api_base": config.api_base,
            "credential_source": config.credential_source,
            "use_bearer_for_custom_base": config.use_bearer_for_custom_base,
        },
        "prompt_sha256": request.prompt_sha256,
        "form_schema_sha256": request.form_schema_sha256,
        "llm_configuration_sha256": request.configuration_sha256,
        "provider_configuration_sha256": config.configuration_sha256,
    }
    return request_audit, provider_kwargs


def _provider_response_content(response: Any) -> str:
    """从 OpenAI 兼容响应中取出第一条文本。"""

    try:
        choices = _value(response, "choices")
        first = choices[0]
        message = _value(first, "message")
        content = _value(message, "content")
    except Exception:
        raise GroundingProviderError(
            "Grounding Provider response has no text content"
        ) from None
    if not isinstance(content, str):
        raise GroundingProviderError("Grounding Provider response content is invalid")
    return content.strip()


def _provider_usage(response: Any) -> GroundingLLMUsage:
    """兼容常见字段名，归一化 Provider 返回的用量。"""

    raw_usage = to_jsonable(_value(response, "usage", default={})) or {}
    if not isinstance(raw_usage, dict):
        raw_usage = {}
    input_tokens = _usage_token(raw_usage, "prompt_tokens", "input_tokens")
    output_tokens = _usage_token(
        raw_usage,
        "completion_tokens",
        "output_tokens",
    )
    total_tokens = _usage_token(raw_usage, "total_tokens")
    if not total_tokens:
        total_tokens = input_tokens + output_tokens
    reasoning_tokens = _usage_token(raw_usage, "reasoning_tokens")
    if not reasoning_tokens:
        for details_name in (
            "completion_tokens_details",
            "output_tokens_details",
        ):
            details = raw_usage.get(details_name)
            if isinstance(details, dict):
                reasoning_tokens = _usage_token(details, "reasoning_tokens")
                if reasoning_tokens:
                    break
    cost = _provider_reported_cost(raw_usage)
    return GroundingLLMUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
        total_tokens=total_tokens,
        cost=cost,
    )


def _provider_reported_cost(raw_usage: Mapping[str, Any]) -> float | None:
    for name in ("cost", "response_cost"):
        value = raw_usage.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        numeric = float(value)
        if math.isfinite(numeric) and numeric >= 0:
            return numeric
    return None


def _usage_token(raw: Mapping[str, Any], *names: str) -> int:
    for name in names:
        value = raw.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if math.isfinite(float(value)) and value >= 0 and int(value) == value:
            return int(value)
    return 0


def _provider_name(response: Any) -> str:
    """尽力读取 LiteLLM 标注的实际 Provider 名称。"""

    hidden = _value(response, "_hidden_params", default={}) or {}
    if isinstance(hidden, Mapping):
        value = hidden.get("custom_llm_provider")
        if isinstance(value, str):
            return value[:128]
    return ""


def _value(value: Any, name: str, *, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _sanitize_audit_value(value: Any) -> Any:
    """递归删除审计数据中的常见敏感字段。"""

    if isinstance(value, Mapping):
        sanitized = {}
        for key, item in value.items():
            normalized = str(key).strip().lower().replace("-", "_")
            if normalized in _SENSITIVE_AUDIT_KEYS:
                continue
            sanitized[str(key)] = _sanitize_audit_value(item)
        return sanitized
    if isinstance(value, (list, tuple)):
        return [_sanitize_audit_value(item) for item in value]
    return value


def _stable_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        preset_canonical_json(value).encode("utf-8")
    ).hexdigest()


def _new_private_audit_path(
    config: GroundingProviderConfig,
    request_sha256: str,
) -> Path:
    """为单次调用生成不冲突的私有审计文件名。"""

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    filename = (
        f"grounding-{timestamp}-{request_sha256[:12]}-{uuid.uuid4().hex[:8]}.json"
    )
    return Path(config.raw_audit_dir) / filename


def _write_private_provider_audit(
    path: Path,
    payload: Mapping[str, Any],
    *,
    api_key: str,
) -> None:
    """用临时文件原子写入权限受限的脱敏审计。"""

    sanitized = _sanitize_audit_value(payload)
    encoded = json.dumps(
        sanitized,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    )
    if api_key and api_key in encoded:
        raise GroundingProviderError("Grounding audit credential check failed")
    # 目录仅当前用户可访问；文件写完后固定为 0600。
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        os.chmod(path.parent, 0o700)
    except OSError:
        pass
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            os.chmod(temporary, 0o600)
            handle.write(encoded)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        if temporary.exists():
            temporary.unlink()


def _private_audit_ref(
    config: GroundingProviderConfig,
    path: Path,
) -> str:
    """优先返回相对项目路径，避免审计结果泄露机器绝对路径。"""

    project_root = Path(config.raw_audit_dir).parents[1]
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return path.name


def _parse_llm_frame_response(
    content: str,
    *,
    observation_text: str,
) -> LLMFrameProposal:
    """严格解析模型 JSON：拒绝重复键、NaN 和多余字段。"""

    if not isinstance(content, str):
        raise LLMFrameUpdateError("transport_format_invalid")
    if len(content) > MAX_LLM_FRAME_RESPONSE_CHARS:
        raise LLMFrameUpdateError("transport_format_invalid")
    try:
        raw = json.loads(
            content,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except LLMFrameUpdateError:
        raise
    except json.JSONDecodeError as exc:
        raise LLMFrameUpdateError("json_invalid") from exc
    except ValueError as exc:
        raise LLMFrameUpdateError("json_invalid") from exc
    if not isinstance(raw, dict):
        raise LLMFrameUpdateError("form_validation_failed")
    proposal = LLMFrameProposal.model_validate(raw)
    _validate_verbatim_mentions(proposal, observation_text)
    return proposal


def _validate_verbatim_mentions(
    proposal: LLMFrameProposal,
    observation_text: str,
) -> None:
    """要求每个 mention 都是当前 Observation 文本中的逐字连续片段。"""

    if not isinstance(observation_text, str):
        raise TypeError("Observation text must be a string")
    slots = (
        *proposal.value_slots,
        *proposal.schema_slots,
        *proposal.operation_slots,
    )
    for slot in slots:
        if not slot.mention or not slot.mention.strip():
            raise LLMFrameUpdateError("mention_validation_failed")
        if slot.mention not in observation_text:
            raise LLMFrameUpdateError("mention_validation_failed")


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    """让 json.loads 遇到重复键时直接失败。"""

    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise LLMFrameUpdateError("duplicate_json_key")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _llm_patch_from_proposal(
    observation: Observation,
    state: RequirementGroundingState,
    proposal: LLMFrameProposal,
    *,
    base_revision: int,
) -> RequirementGroundingPatch:
    """把已校验的 LLM Frame 转成原子 Patch。"""

    existing = _slot_index(state)
    evidence_id = f"evidence.{observation.observation_id}"
    additions: list[GroundingSlot] = []
    updates: list[GroundingSlot] = []

    # 三类提案统一走同一套稳定 ID、更新和证据逻辑。
    typed_proposals: list[tuple[str, KernelModel]] = [
        *(('value', item) for item in proposal.value_slots),
        *(('schema', item) for item in proposal.schema_slots),
        *(('operation', item) for item in proposal.operation_slots),
    ]

    if observation.observation_type == "user_answer":
        return _llm_reconciled_patch(
            observation,
            state,
            typed_proposals,
            base_revision=base_revision,
            allow_new_slots=False,
        )
    if observation.observation_type == "user_query" and observation.phase == 2:
        return _llm_reconciled_patch(
            observation,
            state,
            typed_proposals,
            base_revision=base_revision,
            allow_new_slots=True,
        )

    # Phase 1 初始问题保留 P4.2 已冻结的、按完整提案生成 Slot ID 的行为。
    for slot_kind, item in typed_proposals:
        slot_id = _llm_slot_id(slot_kind, item)
        current = existing.get(slot_id)
        if current is not None and current.lifecycle != "active":
            continue
        slot = _llm_slot_from_proposal(
            slot_kind,
            item,
            slot_id=slot_id,
            evidence_id=evidence_id,
            observation=observation,
            current=current,
        )
        if current is None:
            additions.append(slot)
        else:
            updates.append(slot)
        existing[slot.slot_id] = slot

    if not additions and not updates:
        return RequirementGroundingPatch(
            patch_id=f"llm-noop.{observation.observation_id}",
            base_revision=base_revision,
            source_observation_ids=(observation.observation_id,),
            ambiguity_additions=(),
            ambiguity_updates=(),
            diagnostics=(
                "P4.2a LLM Frame contract returned no provisional changes",
            ),
        )

    evidence = GroundingEvidence(
        evidence_id=evidence_id,
        observation_id=observation.observation_id,
        source_type=f"llm_{observation.observation_type}",
        phase=observation.phase,
        summary=observation.summary,
        raw_digest=observation.raw_digest,
        raw_log_ref=observation.raw_log_ref,
        sequence=observation.sequence,
        timestamp=None,
    )
    return RequirementGroundingPatch(
        patch_id=f"llm.{observation.observation_id}",
        base_revision=base_revision,
        source_observation_ids=(observation.observation_id,),
        slot_additions=tuple(additions),
        slot_updates=tuple(updates),
        ambiguity_additions=(),
        ambiguity_updates=(),
        evidence_additions=(evidence,),
        diagnostics=(
            "P4.2a provisional LLM Frame; no ambiguity or schema binding",
        ),
    )


def _llm_reconciled_patch(
    observation: Observation,
    state: RequirementGroundingState,
    typed_proposals: list[tuple[str, KernelModel]],
    *,
    base_revision: int,
    allow_new_slots: bool,
) -> RequirementGroundingPatch:
    """按 slot_kind + slot_role 原子匹配 user_answer 或 Phase-2 follow-up。"""

    proposal_counts = Counter(
        (slot_kind, item.slot_role) for slot_kind, item in typed_proposals
    )
    duplicate_keys = sorted(
        key for key, count in proposal_counts.items() if count != 1
    )
    if duplicate_keys:
        return _llm_reconciliation_noop(
            observation,
            base_revision,
            "duplicate proposal key",
        )

    all_slots = tuple(_slot_index(state).values())
    active_by_key: dict[tuple[str, str], list[GroundingSlot]] = {}
    for slot in all_slots:
        if slot.lifecycle == "active":
            active_by_key.setdefault((slot.slot_kind, slot.slot_role), []).append(slot)

    for key in proposal_counts:
        matches = active_by_key.get(key, [])
        if len(matches) > 1:
            return _llm_reconciliation_noop(
                observation,
                base_revision,
                "multiple active Slot matches",
            )
        if not allow_new_slots and len(matches) != 1:
            return _llm_reconciliation_noop(
                observation,
                base_revision,
                "user_answer requires exactly one active Slot match",
            )

    evidence_id = f"evidence.{observation.observation_id}"
    additions: list[GroundingSlot] = []
    updates: list[GroundingSlot] = []
    existing_ids = {slot.slot_id for slot in all_slots}
    for slot_kind, item in typed_proposals:
        key = (slot_kind, item.slot_role)
        matches = active_by_key.get(key, [])
        current = matches[0] if matches else None
        slot_id = (
            current.slot_id
            if current is not None
            else _llm_unique_addition_id(
                slot_kind,
                item,
                observation=observation,
                existing_ids=existing_ids,
            )
        )
        slot = _llm_slot_from_proposal(
            slot_kind,
            item,
            slot_id=slot_id,
            evidence_id=evidence_id,
            observation=observation,
            current=current,
        )
        if current is None:
            additions.append(slot)
            existing_ids.add(slot.slot_id)
        else:
            updates.append(slot)

    if not additions and not updates:
        return _llm_reconciliation_noop(
            observation,
            base_revision,
            "proposal contained no Slot changes",
        )

    evidence = GroundingEvidence(
        evidence_id=evidence_id,
        observation_id=observation.observation_id,
        source_type=f"llm_{observation.observation_type}",
        phase=observation.phase,
        summary=observation.summary,
        raw_digest=observation.raw_digest,
        raw_log_ref=observation.raw_log_ref,
        sequence=observation.sequence,
        timestamp=None,
    )
    return RequirementGroundingPatch(
        patch_id=f"llm-reconcile.{observation.observation_id}",
        base_revision=base_revision,
        source_observation_ids=(observation.observation_id,),
        slot_additions=tuple(additions),
        slot_updates=tuple(updates),
        ambiguity_additions=(),
        ambiguity_updates=(),
        evidence_additions=(evidence,),
        diagnostics=(
            "P4.3a deterministic Slot reconciliation; atomic semantic update",
        ),
    )


def _llm_reconciliation_noop(
    observation: Observation,
    base_revision: int,
    reason: str,
) -> RequirementGroundingPatch:
    """用有界诊断拒绝整次语义更新，同时仍把 Observation 标为已处理。"""

    return RequirementGroundingPatch(
        patch_id=f"llm-reconcile-noop.{observation.observation_id}",
        base_revision=base_revision,
        source_observation_ids=(observation.observation_id,),
        diagnostics=(f"P4.3a reconciliation no-op: {reason}"[:512],),
    )


def _llm_unique_addition_id(
    slot_kind: str,
    proposal: KernelModel,
    *,
    observation: Observation,
    existing_ids: set[str],
) -> str:
    """为 Phase-2 新逻辑 Slot 生成稳定 ID，并避免碰撞旧生命周期。"""

    candidate = _llm_slot_id(slot_kind, proposal)
    if candidate not in existing_ids:
        return candidate
    identity = {
        "slot_kind": slot_kind,
        "proposal": proposal.model_dump(mode="json"),
        "observation_id": observation.observation_id,
    }
    digest = hashlib.sha256(
        preset_canonical_json(identity).encode("utf-8")
    ).hexdigest()
    return f"slot.llm.{slot_kind}.{digest[:24]}"


def _llm_slot_id(slot_kind: str, proposal: KernelModel) -> str:
    """根据槽位类型和完整提案内容生成稳定 ID。"""

    identity = {
        "slot_kind": slot_kind,
        "proposal": proposal.model_dump(mode="json"),
    }
    digest = hashlib.sha256(
        preset_canonical_json(identity).encode("utf-8")
    ).hexdigest()
    return f"slot.llm.{slot_kind}.{digest[:24]}"


def _llm_slot_from_proposal(
    slot_kind: str,
    proposal: KernelModel,
    *,
    slot_id: str,
    evidence_id: str,
    observation: Observation,
    current: GroundingSlot | None,
) -> GroundingSlot:
    """把 LLM 提案新建或合并成对应类型的槽位。"""

    evidence_refs = tuple(
        dict.fromkeys((*tuple(getattr(current, "evidence_refs", ())), evidence_id))
    )
    common = {
        "slot_id": slot_id,
        "slot_role": proposal.slot_role,
        "mention": proposal.mention,
        "current_interpretation": proposal.interpretation,
        "grounding_status": "hypothesized",
        "evidence_refs": evidence_refs,
        "ambiguity_refs": (),
        "origin": "llm_provisional",
        "lifecycle": "active",
        "introduced_in_phase": getattr(
            current,
            "introduced_in_phase",
            observation.phase,
        ),
        "last_updated_phase": observation.phase,
        "sequence": observation.sequence,
    }
    if slot_kind == "value":
        return ValueSlot(**common, value_type=proposal.value_type)
    if slot_kind == "schema":
        return SchemaSlot(
            **common,
            binding_type="unknown",
            bound_identifier=None,
        )
    if slot_kind == "operation":
        return OperationSlot(
            **common,
            operation_type=proposal.operation_type,
            parameters=proposal.parameters,
        )
    raise ValueError(f"unsupported LLM proposal slot kind: {slot_kind}")
