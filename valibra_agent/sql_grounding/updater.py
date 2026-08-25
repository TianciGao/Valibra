"""Strict SQL Grounding updater and opt-in LiteLLM Provider adapter.
Structure
  ↓
Mapping
  ↓
Knowledge
  ↓
Check Provider
  ↓
┌─ status=complete
│    → freeze 4D State
│    → 进入 Main
│
├─ status=incomplete + next_tool
│    → 调 1 个 Official tool
│    → 把结果记入 phase-local context
│    → 再调用一次 Check Provider
│    → 循环
│
└─ status=incomplete + next_tool=null
     → terminal incomplete
     → fail-closed
     → 不进 Main

"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import math
import os
import re
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Protocol, TypeAlias
from urllib.parse import urlsplit

from pydantic import Field, ValidationError, model_validator

from shared.audit import to_jsonable
from shared.model_presets import load_model_preset

from valibra_agent.sql_grounding.models import (
    ContractModel,
    GroundingCheckResponse,
    GroundingLLMResponse,
    GroundingRuntime,
    KnowledgeGroundingResponse,
    MappingGroundingResponse,
    StageGroundingResponse,
    StructureGroundingResponse,
    UserClarificationRecord,
    canonical_json,
)
from valibra_agent.sql_grounding.observations import (
    SQLGroundingObservation,
    observation_for_updater,
)
from valibra_agent.sql_grounding.telemetry import (
    GroundingLLMTelemetry,
    GroundingProviderAttemptTelemetry,
    GroundingTokenUsage,
)

MAX_GROUNDING_REQUEST_CHARS = 262_144
MAX_GROUNDING_RESPONSE_CHARS = 65_536
DEFAULT_GROUNDING_TIMEOUT_SECONDS = 600.0
DEFAULT_GROUNDING_MAX_CALLS_PER_TASK = 32
MAX_CHECK_PHASE_LOCAL_CALLS = 32
MAX_CHECK_PHASE_LOCAL_CLARIFICATIONS = 16
GroundingCallKind: TypeAlias = Literal[
    "structure",
    "mapping",
    "knowledge",
    "check",
]
SQL_GROUNDING_STAGE_MAX_TOKENS: dict[GroundingCallKind, int] = {
    "structure": 12_288,
    "mapping": 32_768,
    "knowledge": 24_576,
    "check": 12_288,
}
SQL_GROUNDING_EXACT_EMPTY_RETRY_STAGES = frozenset(
    {"structure", "mapping", "knowledge", "check"}
)
SQL_GROUNDING_MAX_IDENTICAL_RETRIES = 1
SQL_GROUNDING_EXACT_EMPTY_RETRY_REASON = "exact_empty_max_token_failure"
GROUNDING_LLM_ENV_NAMES = (
    "GROUNDING_UPDATER_MODE",
    "GROUNDING_MODEL_PRESET",
    "GROUNDING_TIMEOUT_SECONDS",
    "GROUNDING_MAX_TOKENS",
    "GROUNDING_MAX_CALLS_PER_TASK",
    "GROUNDING_PROMPT_SHA256",
)
_SENSITIVE_AUDIT_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "credential",
        "credentials",
        "password",
        "secret",
        "token",
    }
)
_BEARER_MARKER_RE = re.compile(r"Authorization\s*:\s*Bearer\b", re.IGNORECASE)
_BEARER_TOKEN_RE = re.compile(
    r"Authorization\s*:\s*Bearer\s+([^\s\"'\\]+)(?=$|\s|[\"'])",
    re.IGNORECASE,
)
_RAW_TOKEN_RE = re.compile(r"^[^\s\"'\\=:\{\}\[\]]+$")
_OPAQUE_RAW_CANDIDATE_RE = re.compile(r"^[A-Za-z0-9._-]{16,}$")
_PRIVATE_KEY_BLOCK_RE = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    re.IGNORECASE,
)

_COMMON_EXPRESSION_RULES = """
所有表名和字段都必须来自输入中的 Official evidence。join_keys 和 targets 必须是
sqlglot 26.16.4 PostgreSQL 方言的精确 canonical expression；每个表达式必须采用
expression.sql(dialect="postgres") 渲染所得的精确词法形式。不能包含 SQL 语句、注释、
分号或未批准函数。JSON 运算符 -> 和 ->> 两侧都必须各有一个 ASCII 空格。
仅用于展示语法的示例：t.c -> 'key' ->> 'leaf'。这个示例只展示格式，不能复制未获
当前 Official evidence 支持的标识符或字面量。
column_mapping.phrase 必须逐字来自 query 或 follow_up。
一个确定字段只输出一个 target；只有同一计算或判断确实同时需要多个字段时，才允许多个
targets。targets 不是候选字段集合，不得把 A/B 候选一起塞入。
只返回裸 JSON 对象，不要 Markdown、说明、reasoning 或额外字段。
""".strip()

STRUCTURE_GROUNDING_PROMPT = (
    """你负责 Structure Grounding。只读取 query、current_state 和原始 get_schema evidence。
填写固定表单 {tables, join_keys}。tables 是高召回但由 DDL 支持的候选表；join_keys 是
由 PK/FK/DDL 支持的 canonical 关联表达式。不要填写字段映射、知识、工具或 SQL。
""".strip()
    + "\n\n"
    + _COMMON_EXPRESSION_RULES
)
MAPPING_GROUNDING_PROMPT = (
    """你负责 Mapping Grounding。读取 query、current_state 和候选表内 column_meanings。
填写固定表单 {tables, join_keys, column_mapping}。优先完成 phrase→确定字段表达式映射；
只有 metadata 明确证明 Structure 有误时，才小范围修正 tables/join_keys。

输出形状必须精确为：
{"tables": ["..."],
 "join_keys": ["..."],
 "column_mapping": [{"phrase": "...", "targets": ["..."]}]}

column_mapping 的字段名必须是 targets，不能是 target。targets 永远是 JSON array；
即使只有一个确定字段，也必须写成 ["table.column"]。多个 targets 只允许表示共同
参与同一计算或判断的字段，不是候选集合。
""".strip()
    + "\n\n"
    + _COMMON_EXPRESSION_RULES
)
KNOWLEDGE_GROUNDING_PROMPT = (
    """你负责 Knowledge Grounding。

输入包括 query / follow_up、current_state、Official knowledge_definitions 和 candidate tables
的 column meanings。

你的任务是：
1. 从 Official knowledge_definitions 中选择当前 Query 真正需要的精确 knowledge；
2. 再用选中的 knowledge 检查并修正 current column_mapping。

重要：
- current column_mapping 只是上一轮的暂定结果，可能是错的，不是选择 knowledge 的依据。
- 先根据 Query 的原意以及 knowledge 的 name / description / definition，判断用户实际要求的
  业务概念；选定精确 knowledge 后，再检查 current column_mapping 是否与它一致。
- 如果 current mapping 与精确 knowledge 冲突，应修正 mapping；不要为了保留 current mapping，
  改选一条能解释它的相似 knowledge。
- 相反、相邻、上游、下游或派生概念都不能代替 Query 真正要求的精确概念。
- 如果没有精确匹配的 knowledge，不要猜或选择最接近项，selected_knowledge_ids 返回 []。
- 多个 knowledge id 只能表示完成 Query 确实同时需要多条规则，不能表示候选项。
- 如果 selected_knowledge_ids 返回 []，column_mapping 必须与输入 current_state.column_mapping
  逐项、逐字、顺序完全一致；不得改字段、JSON path、组合表达式或公式。
- 如果 selected_knowledge_ids 非空，只能修改被选中的 Official knowledge 明确、直接支持修正的
  phrase；其他 phrase 的 mapping 必须与输入 current_state.column_mapping 原样保持一致。不得借一条
  knowledge 顺手修改它没有直接提供依据的其他业务概念。
- 如果精确 knowledge 证明字段 A 错、字段 B 对，可以把 column_mapping 从 A 修正为 B；B 必须
  有当前 candidate-table column meanings 支持。

只返回：
{"column_mapping": [{"phrase": "...", "targets": ["..."]}],
 "selected_knowledge_ids": [1, 2]}

规则：column_mapping 返回修正后的完整当前 mapping；selected_knowledge_ids 必须是 JSON array，
ID 必须来自当前 Official knowledge_definitions，没有需要时返回 []；不要复制 definition，不要
生成 knowledge kind，不返回 tables、join_keys 或其他字段，不生成 SQL，不输出解释、reasoning、
Markdown 或额外文字。
""".strip()
    + "\n\n"
    + _COMMON_EXPRESSION_RULES
)
CHECK_GROUNDING_PROMPT = (
    """你负责 Grounding Check。

输入包括：
- query / follow_up
- current_state
- 最新一次补充证据或用户回答（如有）
- previous_official_calls：当前 phase 已真实执行的 Check Official tool name、canonical arguments
  和 request digest；不包含 raw tool result
- answered_clarifications：当前 phase 已真实完成的 exact question / exact answer

previous_official_calls 和 answered_clarifications 都是只读、State 外的 phase-local context：
- previous_official_calls 中已经出现的 exact tool + arguments 不得再次选择；不得通过无意义改写参数
  绕过 duplicate guard。若没有新的合法 evidence direction，返回 terminal incomplete。
- answered_clarifications 可用于 complete / incomplete 判断；即使之后又执行了 Official Check tool，
  其中的回答仍然有效。不得把它们复制、改写或概括进 tables、join_keys、column_mapping 或
  domain_knowledge。

你的任务是检查：当前 State 是否已经有足够证据完成 Query。

0. Clarification Scope Gate（最高优先级）
- 只要存在 latest_user_answer（以及后续 Check tool turn 中保留的 answered_clarifications），
  在检查其他 complete 条件之前，必须先判断该回答属于以下哪一类：
  A. 窄澄清：回答只为 current_state 中已经存在的 mapped concept 补充具体 literal、threshold
  或 formula 参数，没有引入新的字段概念、predicate、业务规则、公式或 AND / OR 组合条件。
  只有这一类回答可以继续下面的完整性检查，并在全部条件通过后允许 complete。
  B. 语义扩展：回答新增或重新定义了完成 Query 所需的字段概念、predicate、业务规则、公式
  或 AND / OR 组合条件。本轮绝对禁止 complete，即使该回答同时解决了上一轮
  missing_information。不得仅依赖 clarification overlay 把这些新增语义交给 Main；如果
  current_state 已包含相关 target，且存在新的合法 Official evidence direction，只选择一个最具体
  的工具，否则返回 incomplete + next_tool = null 并 terminal incomplete。不得使用 execute_sql、
  information_schema 或其他数据库探索重新做 Mapping。
  C. 未解决：回答模糊、拒绝、不知道、不确定、out of scope 或与缺口无关。必须 incomplete；
  不得生成或猜测任何缺失语义。
- B / C 的 incomplete 规则高于所有“回答已经解决上一轮缺口即可 complete”的规则。
- Clarification overlay 只是为已有 Grounding 补充参数的通道，不是替代新 Grounding 语义的通道。

返回 complete 前必须先通过 Gate 0（如有用户回答），再逐项通过以下五项检查：

1. 业务规则完整性
- 如果 Query 需要派生指标、计算公式、阈值或业务判断规则，当前 State 必须包含完成 SQL 所需的
  精确规则及其组合方式。只有相关字段不够；知道涉及 A、B 两个字段，不等于已经知道 A、B
  应如何组合计算。
- 缺少精确规则时必须 incomplete，missing_information 要写明缺少的公式、阈值或判断规则，
  并只选择一个最相关的补证据动作。不得自行补公式或猜规则。

2. Query 与 State 的语义一致性
- 逐项检查 Query / follow_up 的关键概念是否由 column_mapping 和 domain_knowledge 中同一语义的
  字段与规则支持。
- 如果 Query 要求的概念与 State 中已有概念明显不同、相反，或只是相邻/派生概念，不得
  complete；missing_information 必须明确指出冲突，并按现有工具规则选择一个最相关的补证据动作。
- 不得为了保留 current mapping 而放行错误语义。

3. 关键 literal 的权威性与相关性
- 只有 Query 所要求的概念或判断确实依赖某个阈值、类别值或条件时，该 literal 才是 complete
  的必要条件；它必须在 current_state 或本轮合法 Official evidence 中有明确、直接、非示例性的依据。
  用户回答只能直接补充 current_state 中已有 mapped concept 的 literal / threshold；如果
  literal 属于回答新引入的概念或 predicate，该回答本身不能使 State complete，必须先按 Gate 0
  的语义扩展规则处理。
- “for example / e.g. / such as / 例如”等措辞中的 literal 只是示例，不能升级为 frozen mandatory
  predicate，不能要求 Main 把它写进 SQL，也不能据此猜测新的阈值。
- 如果 Query 明确要求某个固定阈值分类，而 State 只有示例性 literal，必须 incomplete，并指出缺少
  authoritative literal；不得采用示例值。
- 如果 Query 只要求排序、最值或返回观测值，并不要求该阈值分类，则示例性 literal 与任务无关：
  不得因为它不具权威性而制造 missing threshold，也不得强制加入对应谓词。Query 已明确给出的
  MAX / MIN / ORDER BY 等操作不要求在 State 中重复成业务规则。
- 没有示例限定词的明确固定 predicate 可以作为 authoritative rule，但仍须先确认 Query 的目标概念
  确实需要该 predicate。

4. entity grain 与 output identity
- 如果 Query 要求返回、比较或排序某个 entity / group，而 measure 来自更细粒度的 event、snapshot
  或 record，只有 State 明确包含该 entity 的 identity / output target，以及 measure 到该 entity 的
  grouping target / 关系时，才能 complete。仅有细粒度 measure 和一条可达 join path 不够。
- 缺少 entity identity 或 grouping grain 时必须 incomplete，missing_information 应明确写出哪个
  entity identity / grouping target 尚未确定。
- Check 不得自动猜 SUM / AVG / MAX 等聚合函数，也不得自动补 mapping。若 entity identity / grouping
  已明确，且 Query 自身已经给出排序、最值或比较语义，则不得仅因 State 没有重复写一个聚合函数
  而制造缺口；其他 completeness 条件满足时可以 complete。

5. 窄澄清是否直接解决上一轮缺口
- 只有 Gate 0 判定为 A（窄澄清）后，才检查回答是否明确、直接提供上一轮缺少的具体
  threshold、formula 参数、literal 或 business-rule 参数。例如缺少 threshold 时，用户明确回答
  “1000 hours”可以解决该缺口；latest_user_answer 仍不等于缺口自动解决。
- 回答解决缺口且第 1–4 项全部通过时，才可返回 complete、missing_information = null、
  next_tool = null；column_mapping 和 domain_knowledge 必须与 current_state 原样保持一致。
- 回答没有解决缺口时必须 incomplete。没有新的合法补证据方向时，next_tool = null 并 terminal
  incomplete；不得为了满足 Form 重复 ask_user，也不得伪造新的 gap 或 tool。
- latest_user_answer 和 answered_clarifications 始终是 State 外的 phase-local clarification evidence，
  不是 Official schema、metadata 或 business knowledge evidence；不得复制、改写或概括进
  tables、join_keys、column_mapping 或 domain_knowledge。

只有 Query 所需的字段、关系、精确业务规则/公式和关键 literal 都已齐全且彼此语义一致时，
才能 complete。Query 本身不需要规则、公式或 literal 时，不得把其缺席凭空当成缺口。

如果已经足够：
- status = "complete"
- missing_information = null
- next_tool = null

如果还不够：
- status = "incomplete"
- missing_information 必须明确写出当前还缺哪一条具体信息。
- 有新的合法补证据方向时，只选择一个最相关的 Official tool，并把 next_tool 填为对应对象。
- 已无新的合法 Official tool 可调用时，next_tool = null。这表示 terminal incomplete：不调用工具、
  不扣 Bird-Coin、不 retry 或 fallback、不修改 State，并且当前 phase fail-closed、不得进入 Main。
- 不要为了“再确认一下”调用工具。

允许的 next_tool 形状只有：

get_column_meaning：
{"tool_name": "get_column_meaning",
 "arguments": {"table_name": "...", "column_name": "..."},
 "user_clarification_request": null}

get_all_external_knowledge_names：
{"tool_name": "get_all_external_knowledge_names",
 "arguments": {},
 "user_clarification_request": null}

get_knowledge_definition：
{"tool_name": "get_knowledge_definition",
 "arguments": {"knowledge_name": "..."},
 "user_clarification_request": null}

execute_sql：
{"tool_name": "execute_sql",
 "arguments": {"sql": "..."},
 "user_clarification_request": null}

ask_user：
{"tool_name": "ask_user",
 "arguments": {"question": "..."},
 "user_clarification_request": {
   "phrase": "...", "kind": "user_intent"}}

重要：
- next_tool 必须是上述对象之一或 null，不能是字符串。
- status = "incomplete" 且 next_tool = null 只用于缺口仍存在、但已无新的合法 Official tool
  可调用的 terminal incomplete；不得用它掩盖可补齐的缺口。
- user_clarification_request 只能位于 next_tool 内部，不能放在顶层。
- ask_user 的 kind 只能精确为 user_intent 或 missing_knowledge。
- ask_user 的 question 只写一次，只能位于 arguments.question；user_clarification_request
  只填 phrase 和 kind，不得重复填写 question。
- user_clarification_request.phrase 必须是 query 或 follow_up 中逐字连续出现的片段。如果对应概念
  已存在于 current_state.column_mapping，优先直接复用该 mapping.phrase，不能添加原 Query 中没有
  的状态、column、identifier 或其他解释性后缀。
- ask_user 只用于用户才能回答的意图或缺失知识，不能询问数据库、schema 或 SQL 实现问题。
- 不允许调用 get_schema、get_all_column_meanings、get_all_knowledge_definitions 或 submit_sql。
- column_mapping 和 domain_knowledge 必须返回修正后的完整当前值；没有新证据需要修改时，必须原样保留，不能随意清空。
- 只能根据当前 State 和最新证据修正 column_mapping / domain_knowledge，不能修改 tables / join_keys。
- 当输入包含 latest_user_answer 时，column_mapping 和 domain_knowledge 必须与 current_state
  完全一致；用户回答只用于 complete / incomplete 判断，绝不能用于修改四维 State。
- 没有足够证据时不要猜。
- Check 只判断并补齐一个最具体的缺口，不重新执行完整 Grounding，也不使用 execute_sql 探索
  业务语义。

只返回下面五个顶层字段：
{"status": "complete 或 incomplete",
 "missing_information": null,
 "next_tool": null,
 "column_mapping": [],
 "domain_knowledge": []}

不要返回其他顶层字段，不要输出解释、reasoning、Markdown 或额外文字。
""".strip()
    + "\n\n"
    + _COMMON_EXPRESSION_RULES
)

SQL_GROUNDING_STAGE_PROMPTS: dict[GroundingCallKind, str] = {
    "structure": STRUCTURE_GROUNDING_PROMPT,
    "mapping": MAPPING_GROUNDING_PROMPT,
    "knowledge": KNOWLEDGE_GROUNDING_PROMPT,
    "check": CHECK_GROUNDING_PROMPT,
}
SQL_GROUNDING_STAGE_PROMPT_SHA256: dict[GroundingCallKind, str] = {
    key: hashlib.sha256(value.encode("utf-8")).hexdigest()
    for key, value in SQL_GROUNDING_STAGE_PROMPTS.items()
}
SQL_GROUNDING_STAGE_FORM_SCHEMAS: dict[GroundingCallKind, dict[str, Any]] = {
    "structure": StructureGroundingResponse.model_json_schema(),
    "mapping": MappingGroundingResponse.model_json_schema(),
    "knowledge": KnowledgeGroundingResponse.model_json_schema(),
    "check": GroundingCheckResponse.model_json_schema(),
}
SQL_GROUNDING_STAGE_FORM_SCHEMA_SHA256: dict[GroundingCallKind, str] = {
    key: hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
    for key, value in SQL_GROUNDING_STAGE_FORM_SCHEMAS.items()
}

# Compatibility names now identify the complete executable 1.3 contract
# bundle, never a hidden fifth Provider form.
SQL_GROUNDING_PROMPT = canonical_json(SQL_GROUNDING_STAGE_PROMPTS)
SQL_GROUNDING_PROMPT_SHA256 = hashlib.sha256(
    SQL_GROUNDING_PROMPT.encode("utf-8")
).hexdigest()
SQL_GROUNDING_FORM_SCHEMA = SQL_GROUNDING_STAGE_FORM_SCHEMAS
SQL_GROUNDING_FORM_SCHEMA_SHA256 = hashlib.sha256(
    canonical_json(SQL_GROUNDING_FORM_SCHEMA).encode("utf-8")
).hexdigest()
SQL_GROUNDING_CONFIGURATION = {
    "check_phase_local_context": {
        "max_answered_clarifications": MAX_CHECK_PHASE_LOCAL_CLARIFICATIONS,
        "max_previous_official_calls": MAX_CHECK_PHASE_LOCAL_CALLS,
        "raw_tool_results": False,
    },
    "stage_form_schema_sha256": SQL_GROUNDING_STAGE_FORM_SCHEMA_SHA256,
    "stage_prompt_sha256": SQL_GROUNDING_STAGE_PROMPT_SHA256,
    "form_schema_sha256": SQL_GROUNDING_FORM_SCHEMA_SHA256,
    "max_calls_per_task": DEFAULT_GROUNDING_MAX_CALLS_PER_TASK,
    "max_request_chars": MAX_GROUNDING_REQUEST_CHARS,
    "max_response_chars": MAX_GROUNDING_RESPONSE_CHARS,
    "provider_execution_policy": {
        "exact_empty_retry_stages": sorted(SQL_GROUNDING_EXACT_EMPTY_RETRY_STAGES),
        "max_identical_retries": SQL_GROUNDING_MAX_IDENTICAL_RETRIES,
        "retry_reason": SQL_GROUNDING_EXACT_EMPTY_RETRY_REASON,
        "sdk_retry_count": 0,
    },
    "prompt_sha256": SQL_GROUNDING_PROMPT_SHA256,
    "stage_max_tokens": SQL_GROUNDING_STAGE_MAX_TOKENS,
    "timeout_seconds": DEFAULT_GROUNDING_TIMEOUT_SECONDS,
    "transport": "bare_json_or_single_json_fence",
}
SQL_GROUNDING_CONFIGURATION_SHA256 = hashlib.sha256(
    canonical_json(SQL_GROUNDING_CONFIGURATION).encode("utf-8")
).hexdigest()

_SINGLE_JSON_FENCE_RE = re.compile(
    r"\A```(?P<label>[A-Za-z]*)\r?\n(?P<body>[\s\S]*?)\r?\n```\Z"
)


class SQLGroundingLLMConfig(ContractModel):
    """Frozen non-secret model configuration for one SQL Grounding call."""

    mode: Literal["llm"] = "llm"
    model_preset: str = Field(min_length=1, max_length=128)
    preset_config: dict[str, Any]
    preset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    timeout_seconds: float = Field(gt=0.0, le=600.0)
    max_tokens: int = Field(ge=1, le=131_072)
    max_calls_per_task: int = Field(ge=1, le=128)
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    form_schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    kernel_configuration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_frozen_contract(self) -> "SQLGroundingLLMConfig":
        if self.timeout_seconds != DEFAULT_GROUNDING_TIMEOUT_SECONDS:
            raise ValueError("GROUNDING_TIMEOUT_SECONDS must equal 600 seconds")
        if self.prompt_sha256 != SQL_GROUNDING_PROMPT_SHA256:
            raise ValueError("GROUNDING_PROMPT_SHA256 mismatch")
        if self.form_schema_sha256 != SQL_GROUNDING_FORM_SCHEMA_SHA256:
            raise ValueError("SQL Grounding form schema SHA mismatch")
        if self.kernel_configuration_sha256 != SQL_GROUNDING_CONFIGURATION_SHA256:
            raise ValueError("SQL Grounding kernel configuration SHA mismatch")
        if self.preset_config.get("max_tokens") != self.max_tokens:
            raise ValueError("GROUNDING_MAX_TOKENS must equal the frozen preset value")
        if self.max_calls_per_task > DEFAULT_GROUNDING_MAX_CALLS_PER_TASK:
            raise ValueError("GROUNDING_MAX_CALLS_PER_TASK cannot exceed 32")
        return self


class SQLGroundingProviderConfig(ContractModel):
    """Validated SQL Grounding connection settings without credential contents."""

    api_base: str = Field(min_length=1, max_length=2048)
    model_id: str = Field(min_length=1, max_length=256)
    credential_source: Literal["direct", "file"]
    api_key_file: str | None = Field(default=None, max_length=2048)
    use_bearer_for_custom_base: bool = False
    retry_count: Literal[0] = 0
    raw_audit_dir: str = Field(min_length=1, max_length=4096)

    @model_validator(mode="after")
    def validate_connection(self) -> "SQLGroundingProviderConfig":
        parsed = urlsplit(self.api_base)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("GROUNDING_API_BASE must be an absolute HTTP(S) URL")
        if self.credential_source == "file" and not self.api_key_file:
            raise ValueError("file credential source requires api_key_file")
        if self.credential_source == "direct" and self.api_key_file is not None:
            raise ValueError("direct credential source cannot retain api_key_file")
        return self


class GroundingLLMRequest(ContractModel):
    """The complete request passed to an injected offline client."""

    prompt: str = Field(min_length=1, max_length=32_768)
    input_json: str = Field(min_length=1, max_length=MAX_GROUNDING_REQUEST_CHARS)
    response_schema: dict[str, Any]
    call_kind: GroundingCallKind = "structure"
    observation_id: str = Field(default="", max_length=256)
    observation_type: str = Field(default="", max_length=64)


class GroundingClientResponse(ContractModel):
    """One Provider result with bounded non-secret audit metadata."""

    content: str
    usage: GroundingTokenUsage = Field(default_factory=GroundingTokenUsage)
    provider_reported_cost: float | None = Field(default=None, ge=0.0)
    model: str = Field(default="", max_length=256)
    provider: str = Field(default="", max_length=128)
    credential_source: Literal["", "direct", "file"] = ""
    request_sha256: str = Field(default="", pattern=r"^(?:|[0-9a-f]{64})$")
    response_sha256: str = Field(default="", pattern=r"^(?:|[0-9a-f]{64})$")
    raw_private_audit_ref: str = Field(default="", max_length=1024)
    provider_may_continue_after_cancel: bool | None = None
    provider_may_bill_after_cancel: bool | None = None
    configured_max_tokens: int = Field(default=0, ge=0, le=131_072)
    attempt_count: int = Field(default=0, ge=0, le=2)
    retry_triggered: bool = False
    retry_trigger_reason: str | None = Field(default=None, max_length=128)
    provider_attempts: tuple[GroundingProviderAttemptTelemetry, ...] = ()
    final_selected_attempt: int | None = Field(default=None, ge=1, le=2)


class AsyncGroundingClient(Protocol):
    async def complete(self, request: GroundingLLMRequest) -> GroundingClientResponse:
        """Return one logical Grounding response without tools."""


class SQLGroundingProviderError(RuntimeError):
    """Sanitized real Provider failure; raw exception text is never retained."""

    def __init__(
        self,
        reason: str,
        *,
        request_sha256: str = "",
        raw_private_audit_ref: str = "",
        attempted: bool = False,
    ) -> None:
        super().__init__(reason)
        self.request_sha256 = request_sha256
        self.raw_private_audit_ref = raw_private_audit_ref
        self.attempted = attempted


class LiteLLMSQLGroundingClient:
    """LiteLLM adapter with one logical request and bounded execution retry."""

    provider_may_continue_after_cancel = True
    provider_may_bill_after_cancel = True

    def __init__(
        self,
        llm_config: SQLGroundingLLMConfig,
        provider_config: SQLGroundingProviderConfig,
        *,
        environment: Mapping[str, str] | None = None,
        completion: Callable[..., Awaitable[Any]] | None = None,
    ) -> None:
        self.llm_config = SQLGroundingLLMConfig.model_validate(llm_config)
        self.config = SQLGroundingProviderConfig.model_validate(provider_config)
        self._environment = environment if environment is not None else os.environ
        self._completion = completion
        self._last_request_sha256 = ""
        self._last_private_audit_ref = ""
        self._last_configured_max_tokens = 0
        self._last_provider_attempts: tuple[GroundingProviderAttemptTelemetry, ...] = ()
        self._last_retry_triggered = False
        self._last_retry_trigger_reason: str | None = None
        self._last_final_selected_attempt: int | None = None
        if self.llm_config.preset_config.get("model") != self.config.model_id:
            raise ValueError("Grounding model preset and Provider model differ")

    async def complete(self, request: GroundingLLMRequest) -> GroundingClientResponse:
        api_key = _read_grounding_api_key(self.config, self._environment)
        request_audit, provider_kwargs = _build_provider_request(
            request,
            self.llm_config,
            self.config,
            api_key=api_key,
        )
        request_sha = _stable_json_sha256(request_audit)
        audit_path = _new_private_audit_path(self.config, request_sha)
        audit_ref = _private_audit_ref(self.config, audit_path)
        self._last_request_sha256 = request_sha
        self._last_private_audit_ref = audit_ref
        completion = self._completion
        if completion is None:
            try:
                import litellm
            except Exception:
                raise SQLGroundingProviderError(
                    "SQL Grounding Provider client unavailable",
                    request_sha256=request_sha,
                    raw_private_audit_ref=audit_ref,
                    attempted=False,
                ) from None
            completion = litellm.acompletion
        started_at = datetime.now(timezone.utc).isoformat()
        started = time.perf_counter()
        observation_id, observation_type = _request_observation_identity(request)
        configured_max_tokens = SQL_GROUNDING_STAGE_MAX_TOKENS[request.call_kind]
        self._last_configured_max_tokens = configured_max_tokens
        self._last_provider_attempts = ()
        self._last_retry_triggered = False
        self._last_retry_trigger_reason = None
        self._last_final_selected_attempt = None
        initial_audit = {
            "schema_version": "1.1",
            "status": "started",
            "started_at": started_at,
            "request": request_audit,
            "request_sha256": request_sha,
            "call_kind": request.call_kind,
            "configured_max_tokens": configured_max_tokens,
            "attempt_count": 0,
            "retry_triggered": False,
            "retry_trigger_reason": None,
            "attempts": [],
            "final_selected_attempt": None,
            "prompt_sha256": SQL_GROUNDING_STAGE_PROMPT_SHA256[request.call_kind],
            "form_schema_sha256": SQL_GROUNDING_STAGE_FORM_SCHEMA_SHA256[
                request.call_kind
            ],
            "configuration_sha256": SQL_GROUNDING_CONFIGURATION_SHA256,
            "model": self.config.model_id,
            "provider": _provider_endpoint_identity(self.config.api_base),
            "credential_source": self.config.credential_source,
            "observation_id": observation_id,
            "observation_type": observation_type,
        }
        try:
            _write_private_provider_audit(
                audit_path,
                initial_audit,
                api_key=api_key,
            )
        except SQLGroundingProviderError:
            raise
        except Exception:
            raise SQLGroundingProviderError(
                "SQL Grounding private request audit preflight failed",
                request_sha256=request_sha,
                raw_private_audit_ref=audit_ref,
                attempted=False,
            ) from None
        attempts: list[GroundingProviderAttemptTelemetry] = []
        raw_attempts: list[dict[str, Any]] = []
        costs: list[float | None] = []
        selected_response: Any = None
        selected_raw_response: Any = None
        selected_response_sha = ""
        selected_content = ""
        retry_triggered = False
        retry_trigger_reason: str | None = None
        provider_kwargs_sha = _stable_provider_kwargs_sha256(provider_kwargs)
        max_attempts = 1 + int(
            request.call_kind in SQL_GROUNDING_EXACT_EMPTY_RETRY_STAGES
        )
        for attempt_number in range(1, max_attempts + 1):
            attempt_kwargs = copy.deepcopy(provider_kwargs)
            if _stable_provider_kwargs_sha256(attempt_kwargs) != provider_kwargs_sha:
                raise SQLGroundingProviderError(
                    "SQL Grounding identical retry request changed",
                    request_sha256=request_sha,
                    raw_private_audit_ref=audit_ref,
                    attempted=bool(attempts),
                )
            attempt_started = time.perf_counter()
            try:
                response = await completion(**attempt_kwargs)
            except asyncio.CancelledError:
                attempt = GroundingProviderAttemptTelemetry(
                    attempt_number=attempt_number,
                    request_sha256=request_sha,
                    latency_ms=_elapsed_ms(attempt_started),
                    status="timed_out",
                    error_type="timeout",
                )
                attempts.append(attempt)
                raw_attempts.append(attempt.model_dump(mode="json"))
                self._set_last_provider_attempts(
                    attempts,
                    retry_triggered=retry_triggered,
                    retry_trigger_reason=retry_trigger_reason,
                )
                _write_private_provider_audit(
                    audit_path,
                    {
                        **initial_audit,
                        "status": "timed_out",
                        "completed_at": datetime.now(timezone.utc).isoformat(),
                        "latency_ms": _elapsed_ms(started),
                        "attempt_count": len(attempts),
                        "retry_triggered": retry_triggered,
                        "retry_trigger_reason": retry_trigger_reason,
                        "attempts": raw_attempts,
                        "provider_may_continue_after_cancel": (
                            self.provider_may_continue_after_cancel
                        ),
                        "provider_may_bill_after_cancel": (
                            self.provider_may_bill_after_cancel
                        ),
                    },
                    api_key=api_key,
                )
                raise
            except Exception as exc:
                error_type = type(exc).__name__[:128]
                attempt = GroundingProviderAttemptTelemetry(
                    attempt_number=attempt_number,
                    request_sha256=request_sha,
                    latency_ms=_elapsed_ms(attempt_started),
                    status="failed",
                    error_type=error_type,
                )
                attempts.append(attempt)
                raw_attempts.append(attempt.model_dump(mode="json"))
                self._set_last_provider_attempts(
                    attempts,
                    retry_triggered=retry_triggered,
                    retry_trigger_reason=retry_trigger_reason,
                )
                _write_private_provider_audit(
                    audit_path,
                    {
                        **initial_audit,
                        "status": "failed",
                        "completed_at": datetime.now(timezone.utc).isoformat(),
                        "latency_ms": _elapsed_ms(started),
                        "attempt_count": len(attempts),
                        "retry_triggered": retry_triggered,
                        "retry_trigger_reason": retry_trigger_reason,
                        "attempts": raw_attempts,
                        "error_type": error_type,
                    },
                    api_key=api_key,
                )
                raise SQLGroundingProviderError(
                    f"SQL Grounding Provider request failed ({error_type})",
                    request_sha256=request_sha,
                    raw_private_audit_ref=audit_ref,
                    attempted=True,
                ) from None

            raw_response = _sanitize_audit_value(to_jsonable(response))
            response_sha = _stable_json_sha256(raw_response)
            usage, cost = _provider_usage(response)
            costs.append(cost)
            finish_reason = _provider_finish_reason(response)
            try:
                content = _provider_response_content(response)
            except SQLGroundingProviderError as exc:
                attempt = GroundingProviderAttemptTelemetry(
                    attempt_number=attempt_number,
                    request_sha256=request_sha,
                    response_sha256=response_sha,
                    finish_reason=finish_reason,
                    usage=usage,
                    latency_ms=_elapsed_ms(attempt_started),
                    content_empty=None,
                    status="invalid_response",
                    error_type="response_content_invalid",
                )
                attempts.append(attempt)
                raw_attempts.append(
                    {
                        **attempt.model_dump(mode="json"),
                        "completion_tokens": usage.output_tokens,
                        "response": raw_response,
                        "provider_reported_cost": cost,
                    }
                )
                self._set_last_provider_attempts(
                    attempts,
                    retry_triggered=retry_triggered,
                    retry_trigger_reason=retry_trigger_reason,
                )
                _write_private_provider_audit(
                    audit_path,
                    {
                        **initial_audit,
                        "status": "failed",
                        "completed_at": datetime.now(timezone.utc).isoformat(),
                        "latency_ms": _elapsed_ms(started),
                        "attempt_count": len(attempts),
                        "retry_triggered": retry_triggered,
                        "retry_trigger_reason": retry_trigger_reason,
                        "attempts": raw_attempts,
                        "error_type": "response_content_invalid",
                    },
                    api_key=api_key,
                )
                raise SQLGroundingProviderError(
                    str(exc),
                    request_sha256=request_sha,
                    raw_private_audit_ref=audit_ref,
                    attempted=True,
                ) from None
            attempt = GroundingProviderAttemptTelemetry(
                attempt_number=attempt_number,
                request_sha256=request_sha,
                response_sha256=response_sha,
                finish_reason=finish_reason,
                usage=usage,
                latency_ms=_elapsed_ms(attempt_started),
                content_empty=not bool(content),
                status="response",
            )
            attempts.append(attempt)
            raw_attempts.append(
                {
                    **attempt.model_dump(mode="json"),
                    "completion_tokens": usage.output_tokens,
                    "response": raw_response,
                    "provider_reported_cost": cost,
                }
            )
            should_retry = (
                attempt_number == 1
                and _is_exact_empty_fuse_hit(
                    call_kind=request.call_kind,
                    content=content,
                    finish_reason=finish_reason,
                    usage=usage,
                    configured_max_tokens=configured_max_tokens,
                )
            )
            if should_retry:
                retry_triggered = True
                retry_trigger_reason = SQL_GROUNDING_EXACT_EMPTY_RETRY_REASON
                self._set_last_provider_attempts(
                    attempts,
                    retry_triggered=True,
                    retry_trigger_reason=retry_trigger_reason,
                )
                _write_private_provider_audit(
                    audit_path,
                    {
                        **initial_audit,
                        "status": "retrying",
                        "attempt_count": len(attempts),
                        "retry_triggered": True,
                        "retry_trigger_reason": retry_trigger_reason,
                        "attempts": raw_attempts,
                    },
                    api_key=api_key,
                )
                continue
            selected_response = response
            selected_raw_response = raw_response
            selected_response_sha = response_sha
            selected_content = content
            self._last_final_selected_attempt = attempt_number
            break

        if selected_response is None:
            raise SQLGroundingProviderError(
                "SQL Grounding Provider did not produce a selected response",
                request_sha256=request_sha,
                raw_private_audit_ref=audit_ref,
                attempted=bool(attempts),
            )
        self._set_last_provider_attempts(
            attempts,
            retry_triggered=retry_triggered,
            retry_trigger_reason=retry_trigger_reason,
            final_selected_attempt=self._last_final_selected_attempt,
        )
        total_usage = _sum_provider_usage(attempts)
        total_cost = _sum_provider_cost(costs)
        _write_private_provider_audit(
            audit_path,
            {
                **initial_audit,
                "status": "succeeded",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "latency_ms": _elapsed_ms(started),
                "attempt_count": len(attempts),
                "retry_triggered": retry_triggered,
                "retry_trigger_reason": retry_trigger_reason,
                "attempts": raw_attempts,
                "final_selected_attempt": self._last_final_selected_attempt,
                "response": selected_raw_response,
                "response_sha256": selected_response_sha,
                "usage": total_usage.model_dump(mode="json"),
                "provider_reported_cost": total_cost,
            },
            api_key=api_key,
        )
        return GroundingClientResponse(
            content=selected_content,
            usage=total_usage,
            provider_reported_cost=total_cost,
            model=self.config.model_id,
            provider=_provider_name(selected_response),
            credential_source=self.config.credential_source,
            request_sha256=request_sha,
            response_sha256=selected_response_sha,
            raw_private_audit_ref=audit_ref,
            provider_may_continue_after_cancel=self.provider_may_continue_after_cancel,
            provider_may_bill_after_cancel=self.provider_may_bill_after_cancel,
            configured_max_tokens=configured_max_tokens,
            attempt_count=len(attempts),
            retry_triggered=retry_triggered,
            retry_trigger_reason=retry_trigger_reason,
            provider_attempts=tuple(attempts),
            final_selected_attempt=self._last_final_selected_attempt,
        )

    def _set_last_provider_attempts(
        self,
        attempts: list[GroundingProviderAttemptTelemetry],
        *,
        retry_triggered: bool,
        retry_trigger_reason: str | None,
        final_selected_attempt: int | None = None,
    ) -> None:
        self._last_provider_attempts = tuple(attempts)
        self._last_retry_triggered = retry_triggered
        self._last_retry_trigger_reason = retry_trigger_reason
        self._last_final_selected_attempt = final_selected_attempt


class GroundingUpdaterResult(ContractModel):
    response: GroundingLLMResponse | StageGroundingResponse
    call_kind: GroundingCallKind | None = None
    telemetry: GroundingLLMTelemetry
    transport_normalization: Literal["none", "single_json_fence"]


class GroundingUpdaterError(RuntimeError):
    """A bounded updater failure with telemetry but no raw response body."""

    def __init__(self, reason: str, telemetry: GroundingLLMTelemetry) -> None:
        super().__init__(reason)
        self.reason = reason
        self.telemetry = telemetry


class SQLGroundingUpdater:
    """Build one bounded request and strictly parse one injected-client response."""

    def __init__(
        self,
        client: AsyncGroundingClient,
    ) -> None:
        self._client = client

    async def propose(
        self,
        runtime: GroundingRuntime,
        observation: SQLGroundingObservation,
        *,
        original_query: str,
        follow_up_query: str | None = None,
        grounding_input: Mapping[str, Any] | None = None,
    ) -> GroundingUpdaterResult:
        """Return a typed proposal without mutating Runtime or Observation."""

        request = _build_request(
            runtime,
            observation,
            original_query=original_query,
            follow_up_query=follow_up_query,
            grounding_input=grounding_input,
        )
        request_sha = hashlib.sha256(canonical_json(request).encode("utf-8")).hexdigest()
        started = time.perf_counter()
        try:
            client_response = await asyncio.wait_for(
                self._client.complete(request),
                timeout=DEFAULT_GROUNDING_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError as exc:
            metadata = _client_telemetry_metadata(self._client)
            telemetry = _telemetry(
                attempted=True,
                status="timed_out",
                latency_ms=_elapsed_ms(started),
                request_sha=(
                    str(getattr(self._client, "_last_request_sha256", ""))
                    or request_sha
                ),
                error_type="timeout",
                call_kind=request.call_kind,
                **metadata,
            )
            raise GroundingUpdaterError("timeout", telemetry) from exc
        except SQLGroundingProviderError as exc:
            metadata = _client_telemetry_metadata(self._client)
            metadata["raw_private_audit_ref"] = exc.raw_private_audit_ref
            telemetry = _telemetry(
                attempted=exc.attempted,
                status="failed" if exc.attempted else "rejected",
                latency_ms=_elapsed_ms(started),
                request_sha=exc.request_sha256 or request_sha,
                error_type=(
                    "provider_error"
                    if exc.attempted
                    else "provider_preflight_error"
                ),
                call_kind=request.call_kind,
                **metadata,
            )
            raise GroundingUpdaterError(
                telemetry.error_type or "provider_error",
                telemetry,
            ) from exc
        except Exception as exc:
            telemetry = _telemetry(
                attempted=True,
                status="failed",
                latency_ms=_elapsed_ms(started),
                request_sha=request_sha,
                error_type="client_error",
                call_kind=request.call_kind,
                **_client_telemetry_metadata(self._client),
            )
            raise GroundingUpdaterError("client_error", telemetry) from exc

        response_sha = client_response.response_sha256 or hashlib.sha256(
            client_response.content.encode("utf-8")
        ).hexdigest()
        effective_request_sha = client_response.request_sha256 or request_sha
        provider_metadata = _response_telemetry_metadata(client_response)
        if len(client_response.content) > MAX_GROUNDING_RESPONSE_CHARS:
            telemetry = _telemetry(
                attempted=True,
                status="failed",
                usage=client_response.usage,
                latency_ms=_elapsed_ms(started),
                request_sha=effective_request_sha,
                response_sha=response_sha,
                error_type="response_too_large",
                call_kind=request.call_kind,
                **provider_metadata,
            )
            raise GroundingUpdaterError("response_too_large", telemetry)
        try:
            payload, normalization = normalize_grounding_transport(
                client_response.content
            )
            try:
                response = parse_grounding_response(
                    payload,
                    call_kind=request.call_kind,
                )
            except _StrictResponseError:
                # Archived offline injected-client fixtures used the retired
                # whole-State form.  They remain replayable, while the real
                # Provider adapter never receives this compatibility path.
                if (
                    isinstance(self._client, LiteLLMSQLGroundingClient)
                    and grounding_input is not None
                ):
                    raise
                response = parse_grounding_response(payload)
        except _StrictResponseError as exc:
            telemetry = _telemetry(
                attempted=True,
                status="failed",
                usage=client_response.usage,
                latency_ms=_elapsed_ms(started),
                request_sha=effective_request_sha,
                response_sha=response_sha,
                error_type=exc.reason,
                call_kind=request.call_kind,
                **provider_metadata,
            )
            raise GroundingUpdaterError(exc.reason, telemetry) from exc

        telemetry = _telemetry(
            attempted=True,
            status="succeeded",
            usage=client_response.usage,
            latency_ms=_elapsed_ms(started),
            request_sha=effective_request_sha,
            response_sha=response_sha,
            call_kind=request.call_kind,
            **provider_metadata,
        )
        return GroundingUpdaterResult(
            response=response,
            call_kind=request.call_kind,
            telemetry=telemetry,
            transport_normalization=normalization,
        )


class _StrictResponseError(ValueError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _build_request(
    runtime: GroundingRuntime,
    observation: SQLGroundingObservation,
    *,
    original_query: str,
    follow_up_query: str | None,
    grounding_input: Mapping[str, Any] | None,
) -> GroundingLLMRequest:
    call_kind: GroundingCallKind = "structure"
    if grounding_input is None:
        input_payload = {
            "current_sql_grounding_state": runtime.grounding_state.model_dump(mode="json"),
            "current_stage": runtime.stage,
            "follow_up": follow_up_query,
            "latest_observation": observation_for_updater(observation),
            "original_query": original_query,
        }
    else:
        input_payload = _validated_bundled_grounding_input(
            grounding_input,
            runtime=runtime,
            original_query=original_query,
            follow_up_query=follow_up_query,
            observation=observation,
        )
        call_kind = classify_grounding_input(
            grounding_input,
            phase=observation.phase,
        )
    input_json = canonical_json(input_payload)
    if len(input_json) > MAX_GROUNDING_REQUEST_CHARS:
        telemetry = _telemetry(
            attempted=False,
            status="rejected",
            error_type="request_too_large",
        )
        raise GroundingUpdaterError("request_too_large", telemetry)
    request = GroundingLLMRequest(
        prompt=SQL_GROUNDING_STAGE_PROMPTS[call_kind],
        input_json=input_json,
        response_schema=SQL_GROUNDING_STAGE_FORM_SCHEMAS[call_kind],
        call_kind=call_kind,
        observation_id=observation.observation_id,
        observation_type=observation.observation_type,
    )
    if len(canonical_json(request)) > MAX_GROUNDING_REQUEST_CHARS:
        telemetry = _telemetry(
            attempted=False,
            status="rejected",
            error_type="request_too_large",
        )
        raise GroundingUpdaterError("request_too_large", telemetry)
    return request


def _validated_bundled_grounding_input(
    grounding_input: Mapping[str, Any],
    *,
    runtime: GroundingRuntime,
    original_query: str,
    follow_up_query: str | None,
    observation: SQLGroundingObservation,
) -> dict[str, Any]:
    """Copy and validate one exact, stage-local Grounding bundle."""

    try:
        call_kind = classify_grounding_input(
            grounding_input,
            phase=observation.phase,
        )
    except (TypeError, ValueError) as exc:
        telemetry = _telemetry(
            attempted=False,
            status="rejected",
            error_type="grounding_bundle_invalid",
        )
        raise GroundingUpdaterError("grounding_bundle_invalid", telemetry) from exc
    payload = dict(grounding_input)
    if payload["query"] != original_query or payload["query"] != original_query.strip():
        telemetry = _telemetry(
            attempted=False,
            status="rejected",
            error_type="grounding_bundle_invalid",
        )
        raise GroundingUpdaterError("grounding_bundle_invalid", telemetry)
    if payload["current_state"] != runtime.grounding_state.model_dump(mode="json"):
        telemetry = _telemetry(
            attempted=False,
            status="rejected",
            error_type="grounding_bundle_invalid",
        )
        raise GroundingUpdaterError("grounding_bundle_invalid", telemetry)
    if observation.phase == 2:
        if (
            not isinstance(follow_up_query, str)
            or not follow_up_query
            or payload["follow_up"] != follow_up_query
            or follow_up_query != follow_up_query.strip()
        ):
            telemetry = _telemetry(
                attempted=False,
                status="rejected",
                error_type="grounding_bundle_invalid",
            )
            raise GroundingUpdaterError("grounding_bundle_invalid", telemetry)
        clarifications = payload["user_clarifications"]
        if not isinstance(clarifications, list):
            telemetry = _telemetry(
                attempted=False,
                status="rejected",
                error_type="grounding_bundle_invalid",
            )
            raise GroundingUpdaterError("grounding_bundle_invalid", telemetry)
        try:
            records = tuple(
                UserClarificationRecord.model_validate(item)
                for item in clarifications
            )
        except ValidationError as exc:
            telemetry = _telemetry(
                attempted=False,
                status="rejected",
                error_type="grounding_bundle_invalid",
            )
            raise GroundingUpdaterError(
                "grounding_bundle_invalid", telemetry
            ) from exc
        if any(item.phase != 1 or item.answer is None for item in records):
            telemetry = _telemetry(
                attempted=False,
                status="rejected",
                error_type="grounding_bundle_invalid",
            )
            raise GroundingUpdaterError("grounding_bundle_invalid", telemetry)
        questions = [item.question for item in records]
        if len(questions) != len(set(questions)):
            telemetry = _telemetry(
                attempted=False,
                status="rejected",
                error_type="grounding_bundle_invalid",
            )
            raise GroundingUpdaterError("grounding_bundle_invalid", telemetry)
    if call_kind == "check":
        previous_calls = payload.get("previous_official_calls")
        answered_clarifications = payload.get("answered_clarifications")
        if (
            not isinstance(previous_calls, list)
            or len(previous_calls) > MAX_CHECK_PHASE_LOCAL_CALLS
            or not isinstance(answered_clarifications, list)
            or len(answered_clarifications)
            > MAX_CHECK_PHASE_LOCAL_CLARIFICATIONS
        ):
            telemetry = _telemetry(
                attempted=False,
                status="rejected",
                error_type="grounding_bundle_invalid",
            )
            raise GroundingUpdaterError("grounding_bundle_invalid", telemetry)
        previous_digests: list[str] = []
        for item in previous_calls:
            if not isinstance(item, dict) or set(item) != {
                "arguments",
                "request_digest",
                "tool_name",
            }:
                telemetry = _telemetry(
                    attempted=False,
                    status="rejected",
                    error_type="grounding_bundle_invalid",
                )
                raise GroundingUpdaterError("grounding_bundle_invalid", telemetry)
            tool_name = item["tool_name"]
            arguments = item["arguments"]
            request_digest = item["request_digest"]
            if (
                not isinstance(tool_name, str)
                or not isinstance(arguments, dict)
                or not re.fullmatch(r"[0-9a-f]{64}", request_digest or "")
            ):
                telemetry = _telemetry(
                    attempted=False,
                    status="rejected",
                    error_type="grounding_bundle_invalid",
                )
                raise GroundingUpdaterError("grounding_bundle_invalid", telemetry)
            expected_digest = hashlib.sha256(
                f"{tool_name}:{canonical_json(arguments)}".encode("utf-8")
            ).hexdigest()
            if request_digest != expected_digest:
                telemetry = _telemetry(
                    attempted=False,
                    status="rejected",
                    error_type="grounding_bundle_invalid",
                )
                raise GroundingUpdaterError("grounding_bundle_invalid", telemetry)
            previous_digests.append(request_digest)
        if len(previous_digests) != len(set(previous_digests)):
            telemetry = _telemetry(
                attempted=False,
                status="rejected",
                error_type="grounding_bundle_invalid",
            )
            raise GroundingUpdaterError("grounding_bundle_invalid", telemetry)
        clarification_questions: list[str] = []
        for item in answered_clarifications:
            if (
                not isinstance(item, dict)
                or set(item) != {"answer", "question"}
                or not isinstance(item["question"], str)
                or not item["question"].strip()
                or not isinstance(item["answer"], str)
                or not item["answer"].strip()
            ):
                telemetry = _telemetry(
                    attempted=False,
                    status="rejected",
                    error_type="grounding_bundle_invalid",
                )
                raise GroundingUpdaterError("grounding_bundle_invalid", telemetry)
            clarification_questions.append(item["question"])
        if len(clarification_questions) != len(set(clarification_questions)):
            telemetry = _telemetry(
                attempted=False,
                status="rejected",
                error_type="grounding_bundle_invalid",
            )
            raise GroundingUpdaterError("grounding_bundle_invalid", telemetry)
        check_fields = {
            name
            for name in ("check_context", "latest_tool", "latest_user_answer")
            if name in payload
        }
        if len(check_fields) != 1:
            telemetry = _telemetry(
                attempted=False,
                status="rejected",
                error_type="grounding_bundle_invalid",
            )
            raise GroundingUpdaterError("grounding_bundle_invalid", telemetry)
    try:
        canonical_json(payload)
    except (TypeError, ValueError, RecursionError) as exc:
        telemetry = _telemetry(
            attempted=False,
            status="rejected",
            error_type="grounding_bundle_invalid",
        )
        raise GroundingUpdaterError("grounding_bundle_invalid", telemetry) from exc
    return payload


_CALL_KIND_EVIDENCE_FIELDS: Mapping[GroundingCallKind, frozenset[str]] = {
    "structure": frozenset({"schema"}),
    "mapping": frozenset({"column_meanings"}),
    "knowledge": frozenset(
        {"knowledge_definitions", "relevant_column_meanings"}
    ),
    "check": frozenset(
        {
            "answered_clarifications",
            "check_context",
            "previous_official_calls",
        }
    ),
}

_CHECK_PHASE_LOCAL_CONTEXT_FIELDS = frozenset(
    {"answered_clarifications", "previous_official_calls"}
)


def classify_grounding_input(
    grounding_input: Mapping[str, Any],
    *,
    phase: Literal[1, 2],
) -> GroundingCallKind:
    """Classify one 1.2 request only by its exact finite field set."""

    if not isinstance(grounding_input, Mapping):
        raise TypeError("Grounding input must be a mapping")
    common = {"query", "current_state"}
    if phase == 2:
        common.update({"follow_up", "user_clarifications"})
    elif phase != 1:
        raise ValueError("Grounding phase must be 1 or 2")
    fields = set(grounding_input)
    matches = [
        call_kind
        for call_kind, evidence_fields in _CALL_KIND_EVIDENCE_FIELDS.items()
        if fields == common | set(evidence_fields)
    ]
    check_suffixes = tuple(
        _CHECK_PHASE_LOCAL_CONTEXT_FIELDS | {latest_field}
        for latest_field in ("check_context", "latest_tool", "latest_user_answer")
    )
    if "check" not in matches and any(
        fields == common | suffix for suffix in check_suffixes
    ):
        matches.append("check")
    if len(matches) != 1:
        raise ValueError("Grounding input has an invalid stage-local field set")
    return matches[0]


def normalize_grounding_transport(
    content: str,
) -> tuple[str, Literal["none", "single_json_fence"]]:
    """Strip only one complete JSON/unlabelled Markdown transport fence."""

    if not isinstance(content, str):
        raise _StrictResponseError("transport_format_invalid")
    stripped = content.strip()
    if "```" not in stripped:
        return content, "none"
    if not stripped.startswith("```"):
        if re.search(r"(?m)^[ \t]*```", stripped) or _contains_unquoted_fence(stripped):
            raise _StrictResponseError("transport_format_invalid")
        return content, "none"
    match = _SINGLE_JSON_FENCE_RE.fullmatch(stripped)
    fence_lines = re.findall(r"(?m)^[ \t]*```", stripped)
    if match is None or len(fence_lines) != 2:
        raise _StrictResponseError("transport_format_invalid")
    label = match.group("label")
    if label and label.lower() != "json":
        raise _StrictResponseError("transport_format_invalid")
    return match.group("body"), "single_json_fence"


def parse_grounding_response(
    payload: str,
    *,
    call_kind: GroundingCallKind | None = None,
) -> GroundingLLMResponse | StageGroundingResponse:
    """Run strict JSON, duplicate-key, finite-value, and Pydantic validation."""

    try:
        raw = json.loads(
            payload,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_non_finite_constant,
        )
    except _StrictResponseError:
        raise
    except (json.JSONDecodeError, ValueError) as exc:
        raise _StrictResponseError("json_invalid") from exc
    if not isinstance(raw, dict):
        raise _StrictResponseError("form_validation_failed")
    try:
        model: type[GroundingLLMResponse] | type[StageGroundingResponse]
        model = {
            "structure": StructureGroundingResponse,
            "mapping": MappingGroundingResponse,
            "knowledge": KnowledgeGroundingResponse,
            "check": GroundingCheckResponse,
            None: GroundingLLMResponse,
        }[call_kind]
        return model.model_validate(raw)
    except ValidationError as exc:
        raise _StrictResponseError("form_validation_failed") from exc


def _contains_unquoted_fence(text: str) -> bool:
    in_string = False
    escaped = False
    index = 0
    while index < len(text):
        character = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
        elif character == '"':
            in_string = True
        elif text.startswith("```", index):
            return True
        index += 1
    return False


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _StrictResponseError("duplicate_json_key")
        result[key] = value
    return result


def _reject_non_finite_constant(_: str) -> None:
    raise ValueError("non-finite JSON constants are forbidden")


def _elapsed_ms(started: float) -> float:
    return max(0.0, (time.perf_counter() - started) * 1000.0)


def requested_sql_grounding_updater_mode(
    environment: Mapping[str, str] | None = None,
) -> str:
    """Return the exact requested mode; empty is the SG3 passthrough default."""

    source = environment if environment is not None else os.environ
    return str(source.get("GROUNDING_UPDATER_MODE", "")).strip()


def load_sql_grounding_llm_config(
    project_root: Path,
    environment: Mapping[str, str] | None = None,
) -> SQLGroundingLLMConfig:
    """Read only explicit ``GROUNDING_*`` model settings without activation."""

    source = environment if environment is not None else os.environ
    missing = [name for name in GROUNDING_LLM_ENV_NAMES if not source.get(name)]
    if missing:
        raise ValueError(f"missing frozen SQL Grounding configuration: {missing}")
    if str(source["GROUNDING_UPDATER_MODE"]).strip() != "llm":
        raise ValueError("GROUNDING_UPDATER_MODE must be exactly 'llm'")
    if str(source["GROUNDING_PROMPT_SHA256"]).strip() != SQL_GROUNDING_PROMPT_SHA256:
        raise ValueError("GROUNDING_PROMPT_SHA256 does not match SQL Grounding prompt")
    preset = load_model_preset(
        project_root,
        str(source["GROUNDING_MODEL_PRESET"]).strip(),
    )
    timeout = _parse_positive_float(
        str(source["GROUNDING_TIMEOUT_SECONDS"]),
        "GROUNDING_TIMEOUT_SECONDS",
    )
    max_tokens = _parse_positive_int(
        str(source["GROUNDING_MAX_TOKENS"]),
        "GROUNDING_MAX_TOKENS",
    )
    max_calls = _parse_positive_int(
        str(source["GROUNDING_MAX_CALLS_PER_TASK"]),
        "GROUNDING_MAX_CALLS_PER_TASK",
    )
    return SQLGroundingLLMConfig(
        model_preset=preset.name,
        preset_config=preset.normalized_config,
        preset_sha256=preset.normalized_sha256,
        timeout_seconds=timeout,
        max_tokens=max_tokens,
        max_calls_per_task=max_calls,
        prompt_sha256=SQL_GROUNDING_PROMPT_SHA256,
        form_schema_sha256=SQL_GROUNDING_FORM_SCHEMA_SHA256,
        kernel_configuration_sha256=SQL_GROUNDING_CONFIGURATION_SHA256,
    )


def load_sql_grounding_provider_config(
    project_root: Path,
    llm_config: SQLGroundingLLMConfig,
    environment: Mapping[str, str] | None = None,
) -> SQLGroundingProviderConfig:
    """Validate an independent Provider endpoint and exactly one credential source."""

    source = environment if environment is not None else os.environ
    api_base = str(source.get("GROUNDING_API_BASE", "")).strip()
    direct_present = bool(str(source.get("GROUNDING_API_KEY", "")).strip())
    key_file_raw = str(source.get("GROUNDING_API_KEY_FILE", "")).strip()
    if not api_base:
        raise ValueError("missing Grounding Provider variable: GROUNDING_API_BASE")
    if direct_present == bool(key_file_raw):
        raise ValueError(
            "configure exactly one of GROUNDING_API_KEY or GROUNDING_API_KEY_FILE"
        )
    credential_source: Literal["direct", "file"]
    resolved_key_file: str | None = None
    if key_file_raw:
        credential_source = "file"
        key_path = Path(key_file_raw).expanduser()
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
    model_id = str(llm_config.preset_config.get("model", "")).strip()
    if not model_id:
        raise ValueError("Grounding model preset has no model ID")
    return SQLGroundingProviderConfig(
        api_base=api_base,
        model_id=model_id,
        credential_source=credential_source,
        api_key_file=resolved_key_file,
        use_bearer_for_custom_base=use_bearer,
        retry_count=0,
        raw_audit_dir=str(
            (project_root / "research-runtime" / "sql-grounding-llm").resolve()
        ),
    )


def build_real_sql_grounding_updater(
    project_root: Path,
    environment: Mapping[str, str] | None = None,
    *,
    completion: Callable[..., Awaitable[Any]] | None = None,
) -> SQLGroundingUpdater:
    """Build the opt-in real updater after both non-secret configs validate."""

    llm_config = load_sql_grounding_llm_config(project_root, environment)
    provider_config = load_sql_grounding_provider_config(
        project_root,
        llm_config,
        environment,
    )
    return SQLGroundingUpdater(
        LiteLLMSQLGroundingClient(
            llm_config,
            provider_config,
            environment=environment,
            completion=completion,
        )
    )


def sql_grounding_provider_health_report(
    project_root: Path,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Return a bounded secret-free requested/effective-mode report."""

    requested = requested_sql_grounding_updater_mode(environment)
    if requested == "":
        return {
            "requested_updater_mode": "passthrough",
            "effective_updater_mode": "passthrough",
            "provider_configuration_valid": True,
            "provider_enabled": False,
            "configuration_error_type": None,
        }
    if requested != "llm":
        return {
            "requested_updater_mode": requested[:64],
            "effective_updater_mode": "invalid",
            "provider_configuration_valid": False,
            "provider_enabled": False,
            "configuration_error_type": "invalid_updater_mode",
        }
    try:
        llm_config = load_sql_grounding_llm_config(project_root, environment)
        load_sql_grounding_provider_config(project_root, llm_config, environment)
    except Exception as exc:
        return {
            "requested_updater_mode": "llm",
            "effective_updater_mode": "invalid",
            "provider_configuration_valid": False,
            "provider_enabled": False,
            "configuration_error_type": type(exc).__name__[:128],
        }
    return {
        "requested_updater_mode": "llm",
        "effective_updater_mode": "llm",
        "provider_configuration_valid": True,
        "provider_enabled": True,
        "configuration_error_type": None,
    }


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


def _read_grounding_api_key(
    config: SQLGroundingProviderConfig,
    environment: Mapping[str, str],
) -> str:
    if config.credential_source == "direct":
        value = str(environment.get("GROUNDING_API_KEY", "")).strip()
        if not value:
            raise SQLGroundingProviderError("Grounding direct credential unavailable")
        return value
    try:
        content = Path(config.api_key_file or "").read_text(encoding="utf-8")
    except OSError:
        raise SQLGroundingProviderError("Grounding credential file unavailable") from None
    return _parse_grounding_credential_file(content)


def _parse_grounding_credential_file(content: str) -> str:
    """Accept one raw token or exactly one Bearer header, and nothing ambiguous."""

    nonempty_lines = [line.strip() for line in content.splitlines() if line.strip()]
    if len(nonempty_lines) == 1 and _RAW_TOKEN_RE.fullmatch(nonempty_lines[0]):
        return nonempty_lines[0]

    stripped = content.strip()
    if (
        not stripped
        or _PRIVATE_KEY_BLOCK_RE.search(content)
        or (stripped.startswith("{") and stripped.endswith("}"))
        or (stripped.startswith("[") and stripped.endswith("]"))
    ):
        raise SQLGroundingProviderError("Grounding credential file format invalid")

    markers = tuple(_BEARER_MARKER_RE.finditer(content))
    matches = tuple(_BEARER_TOKEN_RE.finditer(content))
    if len(markers) != 1 or len(matches) != 1:
        raise SQLGroundingProviderError("Grounding credential file format invalid")

    bearer_line_index = content[: matches[0].start()].count("\n")
    for index, raw_line in enumerate(content.splitlines()):
        if index == bearer_line_index:
            continue
        candidate = raw_line.strip()
        if (
            _OPAQUE_RAW_CANDIDATE_RE.fullmatch(candidate)
            and any(character.isdigit() for character in candidate)
            and any(character in "._-" for character in candidate)
        ):
            raise SQLGroundingProviderError("Grounding credential file format invalid")
    return matches[0].group(1)


def _build_provider_request(
    request: GroundingLLMRequest,
    llm_config: SQLGroundingLLMConfig,
    provider_config: SQLGroundingProviderConfig,
    *,
    api_key: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if request.prompt != SQL_GROUNDING_STAGE_PROMPTS[request.call_kind]:
        raise SQLGroundingProviderError("SQL Grounding prompt mismatch")
    if request.response_schema != SQL_GROUNDING_STAGE_FORM_SCHEMAS[request.call_kind]:
        raise SQLGroundingProviderError("SQL Grounding form schema mismatch")
    messages = [
        {"role": "system", "content": request.prompt},
        {"role": "user", "content": request.input_json},
    ]
    # The Provider transport guarantees only one JSON object.  The frozen
    # Pydantic form and SQL Grounding semantic validators remain the sole
    # authoritative shape and business-state gates after transport parsing.
    response_format = {"type": "json_object"}
    preset = llm_config.preset_config
    generation: dict[str, Any] = {
        "temperature": preset.get("temperature", 0.0),
        "max_tokens": SQL_GROUNDING_STAGE_MAX_TOKENS[request.call_kind],
    }
    if "top_p" in preset:
        generation["top_p"] = preset["top_p"]
    extra_body: dict[str, Any] = {}
    if "thinking" in preset:
        extra_body["thinking"] = preset["thinking"]
    if "reasoning_effort" in preset:
        extra_body["reasoning_effort"] = preset["reasoning_effort"]
    provider_kwargs: dict[str, Any] = {
        "model": provider_config.model_id,
        "messages": messages,
        **generation,
        "stream": False,
        "timeout": llm_config.timeout_seconds,
        "num_retries": 0,
        "max_retries": 0,
        "api_base": provider_config.api_base,
        "api_key": api_key,
        "response_format": response_format,
    }
    if extra_body:
        provider_kwargs["extra_body"] = extra_body
    if provider_config.use_bearer_for_custom_base:
        provider_kwargs["use_bearer_for_custom_base"] = True
    request_audit = {
        "model": provider_config.model_id,
        "messages": messages,
        "generation": generation,
        "extra_body": extra_body,
        "timeout_seconds": llm_config.timeout_seconds,
        "provider_retry_count": 0,
        "tools": [],
        "tool_choice": None,
        "response_format": response_format,
        "connection": {
            "api_base": provider_config.api_base,
            "credential_source": provider_config.credential_source,
            "use_bearer_for_custom_base": (
                provider_config.use_bearer_for_custom_base
            ),
        },
        "call_kind": request.call_kind,
        "prompt_sha256": SQL_GROUNDING_STAGE_PROMPT_SHA256[request.call_kind],
        "form_schema_sha256": SQL_GROUNDING_STAGE_FORM_SCHEMA_SHA256[
            request.call_kind
        ],
        "kernel_configuration_sha256": SQL_GROUNDING_CONFIGURATION_SHA256,
        "model_preset": llm_config.model_preset,
        "model_preset_sha256": llm_config.preset_sha256,
    }
    return request_audit, provider_kwargs


def _request_observation_identity(request: GroundingLLMRequest) -> tuple[str, str]:
    """Extract only bounded audit identity from the internally built input."""

    observation_id = request.observation_id
    observation_type = request.observation_type
    if not observation_id or not observation_type:
        try:
            payload = json.loads(request.input_json)
            latest = payload["latest_observation"]
            observation_id = latest["observation_id"]
            observation_type = latest["observation_type"]
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise SQLGroundingProviderError(
                "SQL Grounding request observation identity invalid"
            ) from exc
    if (
        not isinstance(observation_id, str)
        or not observation_id
        or len(observation_id) > 256
        or not isinstance(observation_type, str)
        or not observation_type
        or len(observation_type) > 64
    ):
        raise SQLGroundingProviderError(
            "SQL Grounding request observation identity invalid"
        )
    return observation_id, observation_type


def _provider_endpoint_identity(api_base: str) -> str:
    parsed = urlsplit(api_base)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise SQLGroundingProviderError("SQL Grounding Provider endpoint invalid")
    return f"{parsed.scheme}://{parsed.netloc}"


def _provider_response_content(response: Any) -> str:
    try:
        choices = _value(response, "choices")
        content = _value(_value(choices[0], "message"), "content")
    except Exception:
        raise SQLGroundingProviderError("Provider response has no text content") from None
    if not isinstance(content, str):
        raise SQLGroundingProviderError("Provider response content invalid")
    return content.strip()


def _provider_finish_reason(response: Any) -> str | None:
    try:
        choices = _value(response, "choices")
        value = _value(choices[0], "finish_reason")
    except Exception:
        return None
    if value is None:
        return None
    text = str(value).strip()
    return text[:128] if text else None


def _provider_usage(response: Any) -> tuple[GroundingTokenUsage, float | None]:
    raw = to_jsonable(_value(response, "usage", default={})) or {}
    if not isinstance(raw, dict):
        raw = {}
    input_tokens = _usage_token(raw, "prompt_tokens", "input_tokens")
    output_tokens = _usage_token(raw, "completion_tokens", "output_tokens")
    reasoning_tokens = _usage_token(raw, "reasoning_tokens")
    if not reasoning_tokens:
        for name in ("completion_tokens_details", "output_tokens_details"):
            details = raw.get(name)
            if isinstance(details, dict):
                reasoning_tokens = _usage_token(details, "reasoning_tokens")
                if reasoning_tokens:
                    break
    reasoning_tokens = min(reasoning_tokens, output_tokens)
    cost = _provider_reported_cost(raw)
    return (
        GroundingTokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
            total_tokens=input_tokens + output_tokens,
        ),
        cost,
    )


def _is_exact_empty_fuse_hit(
    *,
    call_kind: GroundingCallKind,
    content: str,
    finish_reason: str | None,
    usage: GroundingTokenUsage,
    configured_max_tokens: int,
) -> bool:
    """Return true only for the frozen, mechanically observable retry trigger."""

    normalized_finish = (
        (finish_reason or "")
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
        .rsplit(".", 1)[-1]
    )
    return (
        call_kind in SQL_GROUNDING_EXACT_EMPTY_RETRY_STAGES
        and content == ""
        and normalized_finish in {"length", "max_tokens"}
        and usage.output_tokens
        in {configured_max_tokens, configured_max_tokens + 1}
    )


def _sum_provider_usage(
    attempts: list[GroundingProviderAttemptTelemetry],
) -> GroundingTokenUsage:
    return GroundingTokenUsage(
        input_tokens=sum(item.usage.input_tokens for item in attempts),
        output_tokens=sum(item.usage.output_tokens for item in attempts),
        reasoning_tokens=sum(item.usage.reasoning_tokens for item in attempts),
        total_tokens=sum(item.usage.total_tokens for item in attempts),
    )


def _sum_provider_cost(costs: list[float | None]) -> float | None:
    if not costs or any(value is None for value in costs):
        return None
    return sum(value for value in costs if value is not None)


def _usage_token(raw: Mapping[str, Any], *names: str) -> int:
    for name in names:
        value = raw.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if math.isfinite(float(value)) and value >= 0 and int(value) == value:
            return int(value)
    return 0


def _provider_reported_cost(raw: Mapping[str, Any]) -> float | None:
    for name in ("cost", "response_cost"):
        value = raw.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        numeric = float(value)
        if math.isfinite(numeric) and numeric >= 0:
            return numeric
    return None


def _provider_name(response: Any) -> str:
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
    if isinstance(value, Mapping):
        sanitized: dict[str, Any] = {}
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
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _stable_provider_kwargs_sha256(provider_kwargs: Mapping[str, Any]) -> str:
    """Hash real call kwargs without retaining the credential in audit/state."""

    non_secret = copy.deepcopy(dict(provider_kwargs))
    non_secret.pop("api_key", None)
    return _stable_json_sha256(non_secret)


def _new_private_audit_path(
    config: SQLGroundingProviderConfig,
    request_sha256: str,
) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return Path(config.raw_audit_dir) / (
        f"sql-grounding-{timestamp}-{request_sha256[:12]}-{uuid.uuid4().hex[:8]}.json"
    )


def _write_private_provider_audit(
    path: Path,
    payload: Mapping[str, Any],
    *,
    api_key: str,
) -> None:
    encoded = json.dumps(
        _sanitize_audit_value(payload),
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    )
    if api_key and api_key in encoded:
        raise SQLGroundingProviderError("private audit credential check failed")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
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


def _private_audit_ref(config: SQLGroundingProviderConfig, path: Path) -> str:
    project_root = Path(config.raw_audit_dir).parents[1]
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return path.name


def _telemetry(
    *,
    attempted: bool,
    status: Literal["succeeded", "failed", "timed_out", "rejected"],
    usage: GroundingTokenUsage | None = None,
    latency_ms: float = 0.0,
    request_sha: str = "",
    response_sha: str = "",
    error_type: str | None = None,
    provider_reported_cost: float | None = None,
    model: str = "",
    provider: str = "",
    credential_source: Literal["", "direct", "file"] = "",
    raw_private_audit_ref: str = "",
    provider_may_continue_after_cancel: bool | None = None,
    provider_may_bill_after_cancel: bool | None = None,
    configured_max_tokens: int = 0,
    attempt_count: int = 0,
    retry_triggered: bool = False,
    retry_trigger_reason: str | None = None,
    provider_attempts: tuple[GroundingProviderAttemptTelemetry, ...] = (),
    final_selected_attempt: int | None = None,
    call_kind: GroundingCallKind | None = None,
) -> GroundingLLMTelemetry:
    return GroundingLLMTelemetry(
        attempted=attempted,
        status=status,
        usage=usage or GroundingTokenUsage(),
        latency_ms=latency_ms,
        timed_out=status == "timed_out",
        error_type=error_type,
        provider_reported_cost=provider_reported_cost,
        model=model,
        provider=provider,
        credential_source=credential_source,
        raw_private_audit_ref=raw_private_audit_ref,
        provider_may_continue_after_cancel=provider_may_continue_after_cancel,
        provider_may_bill_after_cancel=provider_may_bill_after_cancel,
        configured_max_tokens=configured_max_tokens,
        attempt_count=attempt_count,
        retry_triggered=retry_triggered,
        retry_trigger_reason=retry_trigger_reason,
        provider_attempts=provider_attempts,
        final_selected_attempt=final_selected_attempt,
        request_sha256=request_sha,
        response_sha256=response_sha,
        prompt_sha256=(
            SQL_GROUNDING_STAGE_PROMPT_SHA256[call_kind]
            if call_kind is not None
            else SQL_GROUNDING_PROMPT_SHA256
        ),
        form_schema_sha256=(
            SQL_GROUNDING_STAGE_FORM_SCHEMA_SHA256[call_kind]
            if call_kind is not None
            else SQL_GROUNDING_FORM_SCHEMA_SHA256
        ),
        configuration_sha256=SQL_GROUNDING_CONFIGURATION_SHA256,
    )


def _response_telemetry_metadata(response: GroundingClientResponse) -> dict[str, Any]:
    return {
        "provider_reported_cost": response.provider_reported_cost,
        "model": response.model,
        "provider": response.provider,
        "credential_source": response.credential_source,
        "raw_private_audit_ref": response.raw_private_audit_ref,
        "provider_may_continue_after_cancel": (
            response.provider_may_continue_after_cancel
        ),
        "provider_may_bill_after_cancel": response.provider_may_bill_after_cancel,
        "configured_max_tokens": response.configured_max_tokens,
        "attempt_count": response.attempt_count,
        "retry_triggered": response.retry_triggered,
        "retry_trigger_reason": response.retry_trigger_reason,
        "provider_attempts": response.provider_attempts,
        "final_selected_attempt": response.final_selected_attempt,
    }


def _client_telemetry_metadata(client: Any) -> dict[str, Any]:
    config = getattr(client, "config", None)
    return {
        "model": str(getattr(config, "model_id", ""))[:256],
        "credential_source": getattr(config, "credential_source", ""),
        "raw_private_audit_ref": str(
            getattr(client, "_last_private_audit_ref", "")
        )[:1024],
        "provider_may_continue_after_cancel": getattr(
            client, "provider_may_continue_after_cancel", None
        ),
        "provider_may_bill_after_cancel": getattr(
            client, "provider_may_bill_after_cancel", None
        ),
        "configured_max_tokens": int(
            getattr(client, "_last_configured_max_tokens", 0)
        ),
        "attempt_count": len(
            tuple(getattr(client, "_last_provider_attempts", ()))
        ),
        "retry_triggered": bool(
            getattr(client, "_last_retry_triggered", False)
        ),
        "retry_trigger_reason": getattr(
            client, "_last_retry_trigger_reason", None
        ),
        "provider_attempts": tuple(
            getattr(client, "_last_provider_attempts", ())
        ),
        "final_selected_attempt": getattr(
            client, "_last_final_selected_attempt", None
        ),
    }
