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
│    ├─ ask_user
│    │    → 回答写入 State-external clarification overlay
│    │    → 在同一 Official phase 重新执行 Structure → Mapping → Knowledge → Check
│    │
│    └─ 其他 Official evidence tool
│         → 把结果记入 phase-local context
│         → 再调用一次 Check Provider
│         → 循环
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
    FinalRegroundingGateResponse,
    GroundingCheckResponse,
    GroundingLLMResponse,
    GroundingRuntime,
    KnowledgeGroundingResponse,
    MappingGroundingResponse,
    StageGroundingResponse,
    StructureGroundingResponse,
    UnresolvedMapping,
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
MAX_P2_CHECK_CUMULATIVE_EVIDENCE_ENTRIES = 4
MAX_P2_CHECK_CUMULATIVE_EVIDENCE_CHARS = 4_096
MAX_P2_CHECK_CUMULATIVE_RESULT_CHARS = 2_048
GroundingCallKind: TypeAlias = Literal[
    "structure",
    "mapping",
    "knowledge",
    "check",
    "final_gate",
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
FINAL_REGROUNDING_GATE_MAX_TOKENS = 12_288
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
所有新生成或修改的表名、字段和 JSON path 必须来自当前输入中的 Official evidence。

join_keys 和 targets 必须是 sqlglot 26.16.4 PostgreSQL 方言的 canonical expression，
并与 expression.sql(dialect="postgres") 的精确词法渲染结果一致。

Mapping-facing SQL target 必须保留 metadata / DDL 支持的 canonical identifier；
内部 normalized / lowercase lookup key 不是 SQL target，不能直接暴露给 State。
如果 canonical target 缺失或存在大小写歧义，不得猜测。

表达式不能包含完整 SQL 语句、注释、分号或未批准函数。
JSON 运算符 -> 和 ->> 两侧都必须各有一个 ASCII 空格。
仅用于展示语法的示例：t.c -> 'key' ->> 'leaf'。这个示例只展示格式，不能复制未获
当前 Official evidence 支持的标识符或字面量。

column_mapping.phrase 必须逐字来自 query 或 follow_up。
user_clarifications 可以帮助解释 phrase 的真实业务含义，但不能成为新的 phrase 文本来源。

一个确定字段只输出一个 target。
只有完成同一 Query concept 确实同时需要多个字段时，才允许多个 targets。
多个 targets 只表示“这些字段都需要”，不表示 SUM / AVG / 加法 / 比率 / 排序 / 优先级
或其他公式。targets 不是候选字段集合。

只返回裸 JSON 对象，不要 Markdown、解释、reasoning、注释或额外字段。
""".strip()

STRUCTURE_GROUNDING_PROMPT = (
    """你负责 Structure Grounding。

输入包括：
- query / follow_up
- current_state
- user_clarifications（如有）
- 原始 get_schema evidence

user_clarifications 是前一 Grounding cycle 已回答、State 外、只读的用户约束。
新的 Grounding cycle 必须继续考虑其中仍然有效的约束，但不得把 clarification 文本复制、
改写或概括进四维 State。

你的任务只有两件事：
1. 选择后续回答当前累计 Query 真正可能需要的候选表。
2. 根据 DDL 中明确的 PK / FK 关系填写 join_keys。

如果 current_state 已有 tables / join_keys，它们是本阶段输出的完整基线。
返回值必须覆盖 Original Query + Follow-up + 仍然有效的 user_clarifications 所定义的累计需求，
不能只返回本轮增量。

已有 tables / join_keys 默认原样保留；只有以下情况才允许最小删除或修改：
- follow_up / user clarification 明确撤回或取代旧需求；
- 当前 DDL evidence 明确证明旧表或连接错误。

选表保持高召回，但新增表必须至少承担一种明确职责：
- Query data：直接承载后续所需数据；
- Result identity：承载 Query 要返回、比较、排序或分组的 entity identity；
- Required bridge：是连接已需要表的必要 DDL bridge。

仅仅 topic-related、DDL-reachable、可能有用，不足以新增表。
如果已有直接且足够的 DDL path，不要无理由同时加入 parallel association / bridge path。

这一阶段不要做 column_mapping、domain knowledge、SQL、工具选择或业务规则推断。

只返回：
{"tables": ["..."], "join_keys": ["..."]}
""".strip()
    + "\n\n"
    + _COMMON_EXPRESSION_RULES
)
MAPPING_GROUNDING_PROMPT = (
    """你负责 Mapping Grounding。

输入包括：
- query / follow_up
- current_state
- user_clarifications（如有）
- 当前候选表范围内的 column_meanings
- unresolved_mappings（本 phase 先前 Mapping 明确留下的 omission；首次 Mapping 为空）

user_clarifications 是 State 外、只读的用户约束。它们可以帮助解释 query / follow_up 中
业务短语的真实含义，但不能直接成为新的 mapping.phrase，也不能自行充当 metadata evidence。

你的任务是为 query / follow_up 中真正需要落到数据库的每个 SQL-relevant concept 明确给出
Mapping 结论：有直接 canonical metadata 支持时写入 column_mapping；当前证据不足时写入
unresolved_mappings。不要用 targets=[]、特殊字符串或虚构 target 表示 omission。

按下面顺序处理：
1. 根据 query / follow_up 和仍然有效的 user_clarifications，确定当前累计任务真正需要的业务概念；
2. 只为能够由完整用户上下文 + 当前 Official metadata 直接、唯一闭合的 SQL-relevant concept
   生成 mapping；判断时必须保留该 phrase 在完整 query / follow_up 中的限定词、对象、动作和
   语义角色，不能只按字段名或描述相似度选择 target；
3. 从 current_state 的 tables / join_keys / column_mapping 以及输入 unresolved_mappings 的完整副本开始；
4. 默认保留已有条目；
5. 新澄清如果重新定义了已有 query phrase，可以基于新的用户意图和 Official metadata
   定向修正该 phrase 的 target；
6. 只有 follow_up / clarification 明确撤回旧需求，或当前 metadata 明确证明已有 mapping 错误时，
   才允许最小删除或修改。

返回值必须覆盖 Original Query + Follow-up 当前累计仍然有效的字段需求，而不是本轮 delta。

column_mapping 规则：
- phrase 必须是 query 或 follow_up 中逐字连续出现的原文片段；
- 判断 phrase 的含义时必须结合完整 Query 上下文，不能只看 phrase 与字段名是否相似；
- 只有当当前 metadata 能直接、唯一支持用户真正要求的业务概念时，才生成 mapping。
  一个简单判断是：如果用 target 的 metadata 含义替换 Query 中这个 concept，
  原问题的业务含义基本不变，也不需要额外加入用户没有表达的业务假设，
  才可以认为是直接映射；
- 如果存在两个或以上业务含义不同的合理解释，而 Query / follow_up /
  user_clarifications 仍无法确定是哪一个，不要任选一个，也不要把候选字段全部输出；
   本轮把这个 phrase 写入 unresolved_mappings，reason=ambiguous_user_intent；
- 对派生 concept，先判断其计算或判定关系的来源。如果 query / follow_up 没有明确给出完成
  该 concept 所需的 operator、operands 和方向，不得根据 metadata、字段名、指标名称或常识
  补全公式；可直接 Ground 的 operands 分别处理，未闭合 concept 写入 unresolved_mappings，
  reason=derived_rule_required；
- 如果 query / follow_up 本身已经明确给出外层 operator、operands 和方向，该外层关系属于
  Query，不构成缺失的 Mapping-level business rule。分别 Ground 这些 operands；不得仅因该
  外层关系创建 derived_rule_required。任一 operand 尚不能直接、唯一 Ground 时，只为该
  operand 保留对应的 unresolved mapping；
- 上一条授权只覆盖 Query 明确表达的外层关系，不授权推断 operand 内部组成、aggregation、
  normalization、scaling、time scope、grain、dedup 或其它未表达的计算关系。指标名称中出现
  ratio / score / index 等词，本身不构成 operator provenance；
- mapping.phrase 可以使用较短的 Query 原文片段，但不能因为缩短 phrase 而丢掉会改变
  业务含义的 modifier。特别是 calculated / adjusted / corrected 等修饰不能被删除后，
  再把剩余的普通 concept 映射到一个 stored metric。
  如果外围上下文只是帮助唯一消歧、并没有改变 concept 本身，则仍可以使用较短 phrase；
- 一个确定字段只输出一个 target。
  只有完成同一个 Query concept 明确需要多个字段时，才允许多个 targets。
  多个 targets 不是候选字段集合，也不代表 SUM / AVG / ratio 或其他公式；
- 如果现有 metadata 不能直接、唯一确定某个 Query concept 对应的字段，
  就不要为这个 phrase 生成 mapping，而要写入 unresolved_mappings。
  不要为了保持 Query coverage 而选择“最接近”的字段；
  后续 Knowledge / Check 会继续处理这个未解决的 concept。

unresolved_mappings 规则：
- 它只记录 Mapping 自己未完成的字段语义映射；不得用它记录 aggregation、grain、threshold、
  literal、predicate、time scope、output identity 或其它仅由 Check 判断的缺口；
- phrase 与 column_mapping.phrase 一样，必须是 query 或 follow_up 中逐字连续出现的原文片段；
- reason 只能是：
  - ambiguous_user_intent：存在 metadata 无法替用户决定的业务含义；
  - no_direct_metadata：当前 scope 内没有直接、唯一的 metadata target；
  - derived_rule_required：该 concept 需要 Official rule / formula，不能由 stored proxy 闭合；
  - scope_insufficient：当前候选表范围不足以提供所需 canonical target；
- reason 只是 routing hint，不是 evidence，不能据此猜 target 或自动认为 concept 已解决；
- 同一 phrase 必须且只能出现在 column_mapping 或 unresolved_mappings 之一；
- 已有 unresolved phrase 默认原样保留。只有本次是携带新 clarification / Official evidence /
  metadata scope 的合法 re-Grounding，且该 exact phrase 现在获得 canonical target 时，Mapping
  才能把它从 unresolved_mappings 移入 column_mapping；同输入随机重跑不得清除 omission。

Canonical evidence 规则：
- 新生成或修改的 target 必须来自 Mapping-facing column_meanings 中的 canonical schema evidence；
- normalized lookup key 只用于内部检索，不能作为 SQL target；
- JSON / JSONB base column与 path key 都必须有 metadata 支持；
- canonical target 缺失或有歧义时 fail-closed。

这一阶段不要猜业务公式、阈值或聚合方式，不选择 knowledge，不生成 SQL。

输出形状必须精确为：
{"tables": ["..."],
 "join_keys": ["..."],
 "column_mapping": [{"phrase": "...", "targets": ["..."]}],
 "unresolved_mappings": [{"phrase": "...", "reason": "no_direct_metadata"}]}
""".strip()
    + "\n\n"
    + _COMMON_EXPRESSION_RULES
)
MAPPING_REGROUNDING_CONSUMPTION_RULES = """第二轮 re-Grounding consumption contract：

本 Prompt 只用于输入包含 regrounding_context，且 decision 为 REGROUND_MAPPING 或
REGROUND_STRUCTURE 的 Mapping Draft。

- regrounding_context 中的 official_knowledge、answered_clarifications 和 check_gap 是触发本次
  re-Grounding 的必要上下文，必须与 query、current_state、column_meanings 一起重新判断旧 Mapping；
- clarification / check_gap 可以定义本轮必须补齐的 Query requirement，但不能充当字段 metadata；
- 如果 answered_clarifications 已经明确、完整地定义了 unresolved_mappings 中一个 exact Query
  phrase 的、用户有权定义的 predicate、classification、filter、entity scope 或组合条件，并且该定义
  所需的每个数据库字段都能由当前 column_meanings 直接、唯一地确定 canonical target，则必须把该
  原 Query phrase 映射到全部必要 canonical targets，并清除它的 omission；
- 上一条只允许 clarification 授权用户意图中的 literal、operator、threshold、boundary 和 AND / OR
  组合关系；SQL identifier 仍只能来自 Official metadata。这些用户条件继续保留在 clarification
  overlay 中，不得写入 targets，也不得从 clarification 创建新的 mapping.phrase；
- “derived / classification concept 不得映射到 raw operands”的限制适用于缺少权威定义或定义不完整
  的情况，不得阻止上一条已经由用户完整定义的 exact unresolved phrase 在合法 re-Grounding 中
  映射其必要字段。用户回答仍不能替代必须由 Official Knowledge 定义的 business formula / derived
  rule；此类情况继续遵守下面的 Official formula consumption contract；
- 如果 exact Official rule / formula 明确列出完成原 Query concept 所需的 operands，必须把其中每个
  在当前 column_meanings 中有 canonical metadata 支持的必要 operand 映射到该原 Query concept；
- 这些多个 targets 只承载公式所需字段，formula 本身仍留在 domain_knowledge，不得写进 target；
- 不得因为 current_state 已有 stored / derived / proxy target，就用它替代 Official formula 明确要求的
  operands；只有同一 Official evidence 明确证明该字段与该公式结果语义等价时才可保留为替代；
- 与本次 invalidation 无关的 cumulative mappings 必须原样保留；缺少 canonical operand target 时
  不得猜测，必须把 exact Query phrase 保留或写入 unresolved_mappings；
- 旧 unresolved phrase 只有在本轮新增 actionable evidence 确实给出 canonical target 时才能清除；
  仅有 check_gap、reason 或要求“重新考虑”不属于 actionable evidence。
""".strip()
MAPPING_REGROUNDING_GROUNDING_PROMPT = (
    MAPPING_GROUNDING_PROMPT
    + "\n\n"
    + MAPPING_REGROUNDING_CONSUMPTION_RULES
)
MAPPING_VALIDATION_CORRECTION_PROMPT = (
    """你只负责修正一份被确定性 Form / State validator 拒绝的 Mapping 输出。

这不是新的 Grounding，也不是重新解释 Query 的机会。输入中的 query、current_state、
column_meanings 与第一次 Mapping 完全相同；mapping_validation_correction 另外给出：
- rejected_mapping：第一次被拒绝的原始 Mapping；
- validation_error：确定性 validator 的精确错误；
- invalid_phrases：唯一允许修正或删除的 phrase。

严格执行：
1. tables 和 join_keys 必须与 rejected_mapping 完全相同；不得重选表或连接；
2. 不在 invalid_phrases 中的 column_mapping 条目必须逐项原样保留；
3. 对 invalid_phrases 中的条目，只能：
   - 按当前 column_meanings 修正非法 target / JSON path / canonical expression；或
   - 没有 metadata-supported 合法 target 时，从 column_mapping 删除，并以完全相同 phrase、
     reason=no_direct_metadata 写入 unresolved_mappings；
4. rejected_mapping 中已有 unresolved_mappings 必须逐项原样保留；不得新增无关 phrase，
   不得查询新 evidence，不得改变 business concept；
5. 修正后的完整输出仍必须满足普通 Mapping Form 和 validator。

只返回与普通 Mapping 完全相同形状的裸 JSON。""".strip()
    + "\n\n"
    + _COMMON_EXPRESSION_RULES
)
KNOWLEDGE_GROUNDING_PROMPT = (
    """你负责 Knowledge Grounding。

输入包括：
- query / follow_up
- current_state
- user_clarifications（如有）
- Official knowledge_definitions
- 当前候选表范围内的 column_meanings
- unresolved_mappings（Mapping 留下的 State 外只读 omission sidecar）

user_clarifications 是 State 外、只读的用户意图证据。它们可以帮助消歧 Query 真正指的业务概念，
但不能自行充当 Official knowledge / metadata，也不能凭用户回答生成新的 business rule、字段事实
或 formula。

unresolved_mappings 只表示 Mapping 尚未完成的字段语义映射。Knowledge 可以用其中的 exact phrase
帮助寻找直接匹配的 Official knowledge，但它是只读 routing context，不是 evidence：
- 不得新增、修改、删除 unresolved_mappings；
- 不得因为选中 knowledge 就自行填写 omission 的 target；
- 不得触发或请求 re-Grounding；
- 找到 direct Official rule 时，仍只通过 selected_knowledge_ids 正常选择该 rule；
- 找不到 direct rule 时，selected_knowledge_ids 可以为 []，omission 仍由 Mapping carrier 原样持有。

unresolved_mappings 的唯一 owner 是 Mapping。Knowledge response 没有这个字段，也不得通过
column_mapping 新增与 omission exact phrase 相同的 mapping 来绕过只读边界。

你的任务严格分两步，顺序不能反：
第一步：选择完成当前累计 Query 真正需要的 Official knowledge。
第二步：只有 selected knowledge 明确证明某条 current mapping 错误时，才做 bounded targeted correction。

选择 Knowledge 前先做 direct-match gate：
“这条 knowledge 是否直接定义 Query / follow_up / user clarification 真正要求的同一个业务概念
和同一个语义角色，并且没有加入 Query 未要求的更窄限定？”

只有明确 YES 才选择。以下都不能替代 direct match：topic related、supporting metric / formula、
upstream / downstream、derived / correlated concept、narrower subtype / specific rule。

不要根据 current column_mapping 反向选择一条能够解释当前 mapping 的相似 knowledge。
如果没有 direct match：selected_knowledge_ids = []。

Official phase 2 中，runtime 默认保留 current_state.domain_knowledge；本轮
selected_knowledge_ids 只负责从当前 Official inventory 追加直接需要的 knowledge，未选择既有 knowledge
不构成删除；既有 knowledge 不得被删除或替换。

preserve-first mapping correction：
- 从 current_state.column_mapping 的完整副本开始；
- selected=[] 时，mapping 必须逐项、逐字、顺序完全不变；
- selected 非空时，只能修改该 knowledge 明确、直接证明错误的 phrase；
- 其他 phrase 必须原样保持；
- replacement target 必须由当前 Mapping-facing canonical column_meanings 直接支持；
- 不得把 lookup key、未支持 JSON path、任意 SQL expression 或 formula 写成 target；
- 如果没有合法 replacement target，保持旧 mapping fail-closed，让 Check 暴露冲突，不得猜替代值。

多个 knowledge id 只表示完成 Query 确实同时需要多条规则，不是候选项。
多个 targets 也只表示多个字段都需要，不表示组合公式。

只返回：
{"column_mapping": [{"phrase": "...", "targets": ["..."]}],
 "selected_knowledge_ids": [1, 2]}
""".strip()
    + "\n\n"
    + _COMMON_EXPRESSION_RULES
)
CHECK_GROUNDING_PROMPT = """你负责 Grounding Check。

你的任务是判断 current_state 加上当前 Official phase 已获得的合法 Official evidence 与 State 外
user_clarifications，是否已经足够支持 Query；如果不足，只补一个最具体缺口。

不要重新执行完整 Grounding。

输入可能包括：
- query / follow_up
- current_state
- user_clarifications（前序 Grounding cycle 已回答、State 外、只读）
- unresolved_mappings（Mapping 留下的 State 外只读 omission sidecar）
- answered_clarifications（当前 Official phase 的 question / answer 只读投影）
- latest_tool（刚执行完的 Official Check tool 及结果，如有）
- previous_official_calls（当前 Official phase 已执行的 tool + canonical arguments）

user_clarifications 只定义用户真实意图；clarification 不是 schema、metadata 或 Official business
knowledge，永远不得复制、改写、摘要或 materialize 到四维 State。Main 会在 State 外单独接收这层
用户意图，不需要把它伪装成 domain_knowledge。

Mapping omission contract：
- unresolved_mappings 只表示 Mapping 尚未完成的字段语义映射；唯一 owner 是 Mapping；
- Check 不得新增、删除、修改 omission，也不得通过 column_mapping 自行填入同一 exact phrase；
- reason 只是 routing hint，不是 Official evidence；
- unresolved_mappings 非空时，本轮 Check 不得返回 complete。

对当前最高优先级 unresolved mapping，Check 只能选择一个动作：
1. 仍有新的合法 Official evidence 可获得：调用一个最具体 Official tool；
2. 缺口属于用户有权决定的业务意图：ask_user，clarification phrase 必须等于该 omission 的 exact phrase；
3. 本轮已经获得新的 actionable clarification / Official evidence，足以让 Mapping 在不同输入下
   重新判断该 omission：status=incomplete、next_tool=null，交 Final Regrounding Gate；
4. 没有合法 evidence direction，用户也不能解决：terminal incomplete。

Check 不得根据名称相似、proxy、raw measure、“最自然解释”或 SQL 默认习惯消除 omission。
只有后续合法 re-Grounding 中的新 Mapping 输出，才能正式清除 unresolved_mappings entry。

0. Clarification routing（输入含 latest_user_answer 时最高优先级）
- clarification_route 只描述当前这一次 Check request 对顶层 latest_user_answer 的即时路由，
  不能从历史调用继承，也不能表示过去曾经采用过的 route；
- 只有当前输入 JSON 顶层明确存在 latest_user_answer 字段时，才视为“本轮刚收到用户回答”。
  user_clarifications、answered_clarifications 以及 regrounding_context.answered_clarifications 都只是
  历史 evidence，绝不等同于 latest_user_answer；
- 本轮必须先返回一个明确 clarification_route，并且 column_mapping / domain_knowledge 必须与
  current_state 完全一致：
  - stay_check：回答只给 current_state 中已有 mapped concept 补 literal、threshold 或参数，没有新增
    concept、predicate、formula、entity、grain 或组合关系；继续本轮完整性检查，可 complete 或继续取证。
  - restart_grounding：回答新增或重新定义 concept、predicate、formula、entity、grain 或组合关系；
    必须 status=incomplete、next_tool=null。Runtime 会在同一 Official phase 启动新的
    Structure → Mapping → Knowledge → Check cycle；本轮 Check 不得一点点修旧 State。
  - terminal：回答模糊、不知道、拒绝或未解决问题；必须 status=incomplete、next_tool=null。
- 如果 latest_user_answer.clarification_event.origin=atomic_draft，产生问题的 private Draft 已经
  rollback，绝不能返回 stay_check 试图继续旧 Draft。回答未拒绝且能继续处理时应返回
  restart_grounding；runtime 会把它交给 Final Gate 并创建全新的 Draft。
- 当前输入顶层没有 latest_user_answer 时，clarification_route 必须为 none；即使输入包含历史
  user_clarifications / answered_clarifications / regrounding_context，或当前是它们触发的新 Mapping /
  Knowledge 之后的 initial Check，也不能返回 stay_check、restart_grounding 或 terminal。
  历史 clarification 中曾有语义扩展，不是永久禁止 complete 的理由；后续新 Grounding 已完整覆盖时
  仍可 status=complete，但此时 clarification_route 必须为 none。

返回 complete 前逐项检查：

1. 业务规则完整性
- Query 如果需要派生指标、公式、阈值或业务判断规则，State 必须有足够精确的 Official rule
  及组合方式；只有相关字段不够。
- Query 本身不需要公式、阈值或 literal 时，不得凭空制造缺口。

1.1 Exact Official formula operand completeness
- 如果 current_state.domain_knowledge 中存在直接定义 Query concept 的 exact Official formula / rule，
  complete 不仅要求该 rule 存在，还要求公式所需的 SQL operands 已得到合法 Grounding；
- 只有以下两种情况之一成立，才可认为该 formula concept 的字段侧完整：
  A. exact formula 明确要求的每个必要 operand 已由 current_state.column_mapping 承载；
  B. Official evidence 明确证明 current mapping 中某个 stored / derived target 与该 exact formula 的
     结果语义等价，因此可以合法替代这些 operands；
- B 的“明确证明”必须由 Official evidence 直接说明该 target 存储 / 表示该 exact formula 的计算结果，
  或者 target 的 Official definition 明确给出与该 formula 相同的必要 operands、关系和计算语义；
- target 名称与 Query / rule 名称相似、column description 使用相近业务词、单位或类型相同、
  target 是 related / derived / precomputed metric，或者根据字段名与上下文推断“很可能等价”，
  均不足以证明 B；没有上述 explicit Official proof 时不得自行推断 equivalence；
- 如果 exact formula 已存在、必要 operands 的 canonical metadata 在当前 table / join scope 内可用，
  但 current mapping 缺少这些 operands且没有 Official equivalence evidence，则 current Mapping 不完整，
  不得 complete，也不得继续为旧 stored target 寻找语义合理化；若无需再调用新的 Official evidence
  tool，应返回 status=incomplete、clarification_route=none、next_tool=null，并在
  missing_information 中明确说明缺失的 operands / invalidation gap，交给 Final Regrounding Gate；
  不在 Check 中自行修改 Mapping。

2. Direct semantic coverage
- 对 Query / follow_up / user_clarifications 当前要求的每个 SQL-relevant concept，检查 State 是否有
  直接支持该 concept 的字段 / rule。
- related metric、proxy、inverse metric、derived score、supporting signal、相邻概念或更窄/更宽概念
  不能单独证明目标 concept 已闭合。
- 一个 State target 如果不能直接替换 Query target 而保持业务含义不变，则该 concept 仍未 Ground。
- 不得为了保留 current mapping 而放行错误语义。

3. literal / predicate 权威性
- 只有 Query 真正依赖的 threshold、类别值、predicate 才是 complete 条件；
- 必须由 Query、仍然有效的 user clarification 或 Official evidence 明确支持；
- “for example / e.g. / such as / 例如”中的成员只是示例，不能自动升级为 mandatory predicate；
- Query 已明确给出的排序、MAX / MIN 等操作不要求在 State 中重复成 business rule。

4. entity grain 与 output identity
- Query 如果要求返回、比较、排序或分组某个 entity / group，而 measure 来自更细粒度 record，
  State 必须明确包含该 entity identity / output target 以及必要 grouping / relation；
- 仅有细粒度 measure 和一条可达 join path 不够；
- 不得自动猜 SUM / AVG / MAX 等聚合规则。

全部通过才 complete。

如果还不够：
- missing_information 只写当前最具体、最高优先级的一条 gap；
- 有新的合法 Official evidence direction 时，只选择一个最具体 next_tool；
- 没有 latest_user_answer 时，Check 只能根据 current_state + 本轮 latest_tool 的一条合法 Official
  evidence 做 bounded repair：
  - 无 latest_tool 或 latest_tool 只是 knowledge names / execute_sql：两项必须原样保留；
  - latest_tool=get_column_meaning：最多修正该证据直接支持的一条 column_mapping，
    domain_knowledge 原样保留；
  - latest_tool=get_knowledge_definition：column_mapping 原样保留；domain_knowledge 只能保留旧值，
    并至多新增这一条 exact Official definition，形状必须是
    {"kind": "business_rule", "content": "<Official definition 原文>"}；
    禁止返回 {"name": "...", "definition": "..."}，禁止改写或总结 definition；
- 不修改 tables / join_keys；
- 如果 gap 来自 clarification 改变 scope / field concept / rule，而新 Grounding cycle 尚未覆盖它，
  不要在 Check 中一点点重做 Structure / Mapping / Knowledge，应 fail-closed。

previous_official_calls 中已经执行过的 exact tool + arguments 不得重复调用，
也不得通过无意义参数改写绕过 duplicate guard。

ask_user 是高成本 Grounding-cycle boundary，不是普通 schema / evidence discovery 工具。
只有同时满足以下条件才允许：
1. 至少存在两个合理的用户业务解释；
2. 不同解释会实质改变 SQL-relevant Grounding；
3. Official evidence 无法替用户决定真实意图。

生成 ask_user.question 前，先检查当前 clarification phrase 下是否还存在从当前 Query、State 与
Official evidence 已经能够明确预见、必须由用户决定且会实质改变 SQL 的未定项。这些未定项可能
包括：
- 业务概念的具体解释或类别；
- threshold、上下界、比较方向；
- 多个用户定义条件之间的 AND / OR / optional 关系。

一次仍然只询问一个 clarification phrase，但应尽量在同一个问题中收齐这个 phrase 下已经能够预见的
必要用户参数，避免用户回答后立即产生同一概念下新的 user-only 缺口。特别是如果回答可能继续使用
high、low、severe、major、multiple、significant、typical 等定性词，应明确要求用户同时给出可执行的
类别、cutoff、boundary 或业务判定标准；如果存在多个条件，还应说明它们的组合关系。

不得要求用户提供可由 Official metadata / knowledge 确定的信息，不得为了“问完整”而猜测 Query
尚未提出的新条件，不得询问数据库、schema、字段名或 SQL 实现。

优先询问最高影响的一处歧义，一次只问一个问题。question 必须使用用户可理解的业务语言，
不得出现 database / table / column / record / schema / SQL / JSON path、raw target 或 table.column
等实现术语。

user_clarification_request.phrase 必须逐字连续来自 query 或 follow_up。如果该 phrase 已在
current_state.column_mapping 中，优先复用 mapping.phrase。

Atomic Draft 中，如果 Mapping-level omission 已全部清除，但仍缺一个依附于已 Ground concept 的
threshold、category、literal 或 predicate，允许把它表达为一次短生命周期的 Check requirement event：
- requirement_type 只能是 threshold、category、literal 或 predicate；
- related_mapping_phrases 必须非空，并逐项精确复用 current_state.column_mapping 中已经存在的 phrase；
- phrase 仍必须是 query 或 follow_up 的逐字连续片段，可以不同于 related mapping phrase；
- 该 event 只描述依附于现有 Mapping 的用户参数，不得用于索取新的字段语义、formula、derived rule、
  grain、aggregation、time scope 或 output identity；
- 如果仍有 unresolved_mappings，不得使用这个 Check-level event 绕过 Mapping omission。

普通 Mapping-level omission clarification 不填写 requirement_type，related_mapping_phrases 为空。
这些字段只提供 typed routing evidence；Check 仍无权创建、删除或修改 Mapping。

user_intent 用于业务对象、类别、范围、解释或用户意图之间的真实歧义。
missing_knowledge 只用于已有 Grounded concept / rule 中仍缺失且必须由用户确定的参数、literal、
选项或范围；不得要求用户提供一条全新的 formula、字段语义、predicate 或完整 business rule
来替代 Official evidence。

一旦返回 ask_user，当前 Grounding cycle 结束；用户回答必须由 runtime 加入 State 外
user_clarifications，并先由下一次 Check 按 Clarification routing 分类。stay_check 留在 Check，
restart_grounding 才启动新的四阶段 Grounding cycle；不得再次询问已经回答的同一 phrase。

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
只有输入中已经存在明确 candidate_sql 时，才允许用于 SQL implementation validation。
如果没有 candidate_sql，execute_sql 不是合法 next_tool。Check 不负责创造、拼接或探索 SQL。

ask_user：
{"tool_name": "ask_user",
 "arguments": {"question": "..."},
 "user_clarification_request": {
   "phrase": "...", "kind": "user_intent",
   "requirement_type": null, "related_mapping_phrases": []}}

Atomic Draft Check-level requirement event 的 ask_user 形状：
{"tool_name": "ask_user",
 "arguments": {"question": "..."},
 "user_clarification_request": {
   "phrase": "...", "kind": "user_intent",
   "requirement_type": "threshold",
   "related_mapping_phrases": ["<exact existing mapping.phrase>"]}}

重要：
- next_tool 必须是上述对象之一或 null，不能是字符串。
- status = "incomplete" 且 next_tool = null 只用于：缺口已无新的合法 Official tool 可调用的
  terminal incomplete；或本轮已有新的 actionable evidence 足以交 Final Gate 重新 Grounding。
  不得用它掩盖仍可由一个合法工具补齐的缺口。
- user_clarification_request 只能位于 next_tool 内部，不能放在顶层。
- ask_user 的 kind 只能精确为 user_intent 或 missing_knowledge。
- ask_user 的 question 只写一次，只能位于 arguments.question；user_clarification_request
  不得重复填写 question。requirement_type 与 related_mapping_phrases 必须同时填写或同时省略；
  仅上述 Atomic Draft Check-level requirement event 可以填写它们。
- user_clarification_request.phrase 必须是 query 或 follow_up 中逐字连续出现的片段。如果对应概念
  已存在于 current_state.column_mapping，优先直接复用该 mapping.phrase，不能添加原 Query 中没有
  的状态、column、identifier 或其他解释性后缀。
- ask_user 只用于用户才能回答的意图或缺失知识，不能询问数据库、schema 或 SQL 实现问题。
- 不允许调用 get_schema、get_all_column_meanings、get_all_knowledge_definitions 或 submit_sql。
- column_mapping 和 domain_knowledge 必须返回修正后的完整当前值；没有上述授权时必须逐项原样保留，
  不能随意新增、改写或清空。
- latest_user_answer / user_clarifications / answered_clarifications 绝不能成为 State 修改依据。
- 没有足够证据时不要猜。
- Check 只判断并补齐一个最具体的缺口，不重新执行完整 Grounding。

只返回下面六个顶层字段：
status、clarification_route、missing_information、next_tool、column_mapping、domain_knowledge。
clarification_route 只能是 none、stay_check、restart_grounding 或 terminal。
column_mapping 和 domain_knowledge 必须始终返回修正后的完整当前值；除非当前值本来为空，
不得用空数组作占位或清空 State。

不要返回其他顶层字段，不要输出解释、reasoning、Markdown 或额外文字。

所有新生成或修改的表名、字段和 JSON path 必须来自当前输入中的 Official evidence。

join_keys 和 targets 必须是 sqlglot 26.16.4 PostgreSQL 方言的 canonical expression，
并与 expression.sql(dialect="postgres") 的精确词法渲染结果一致。

Mapping-facing SQL target 必须保留 metadata / DDL 支持的 canonical identifier；
内部 normalized / lowercase lookup key 不是 SQL target，不能直接暴露给 State。
如果 canonical target 缺失或存在大小写歧义，不得猜测。

表达式不能包含完整 SQL 语句、注释、分号或未批准函数。
JSON 运算符 -> 和 ->> 两侧都必须各有一个 ASCII 空格。
仅用于展示语法的示例：t.c -> 'key' ->> 'leaf'。这个示例只展示格式，不能复制未获
当前 Official evidence 支持的标识符或字面量。

column_mapping.phrase 必须逐字来自 query 或 follow_up。
user_clarifications 可以帮助解释 phrase 的真实业务含义，但不能成为新的 phrase 文本来源。

一个确定字段只输出一个 target。
只有完成同一 Query concept 确实同时需要多个字段时，才允许多个 targets。
多个 targets 只表示“这些字段都需要”，不表示 SUM / AVG / 加法 / 比率 / 排序 / 优先级
或其他公式。targets 不是候选字段集合。

只返回裸 JSON 对象，不要 Markdown、解释、reasoning、注释或额外字段。""".strip()
FINAL_REGROUNDING_GATE_PROMPT = """你是本轮 Grounding 结束后的薄 invalidation Gate。

Structure、Mapping、Knowledge、ask_user 和 Check 已经完成本轮的信息收集。你不重新做这些阶段，
不修 State，不选工具，也不提出新的用户问题。你只比较：
- 当前正式 4D State；
- 本轮得到的 clarification、Official knowledge/evidence 和最终 Check 结果；
- 当前 schema / metadata 证据。

只判断新信息是否使当前 State 失效，以及最早需要从哪里重新 Grounding：

Mapping owner transition（先于下面四种路由判断）：
- final_gate_context.mapping_owner_transition 是 runtime 从当前 Mapping omission sidecar 生成的只读
  lifecycle 事实；它不替 Check 判断公式、operands 或业务语义是否充分；
- 如果其中列出的 existing omission 仍由 Mapping 独占清除权限，actionable_evidence_changed_since_mapping
  为 true，并且 final_check 已明确说明当前合法 evidence 足以让 Mapping 重新判断并清除该 omission，
  则必须选择 REGROUND_MAPPING。即使公式与 operands 在语义上已经齐全，仍存在的 omission 也表示
  lifecycle 尚未完成；不得把 semantic completeness 当成 NO_REGROUND；
- 只有 omission 非空本身不构成 re-Grounding 理由。如果 final_check 仍在报告缺少 rule、operand、
  denominator、grain、dedup、用户参数或其它 evidence，不得自行推导它已经可清除，也不得仅凭该
  lifecycle carrier 选择 REGROUND_MAPPING；
- Gate 不重新验证公式或选择 operands，只消费 final_check 对 actionable evidence 的结论并完成
  owner routing。相同 omission、evidence 与 Mapping input 的重复 transition 由 runtime bounded terminal。

1. NO_REGROUND
   新信息没有使 tables / join_keys / column_mapping 失效。已有 mapped concept 的 literal、threshold、
   boundary、排序方向或其它窄参数属于 clarification overlay，不触发 re-Grounding。

2. REGROUND_MAPPING
   新信息引入或重定义 SQL-relevant concept、operand、predicate、formula input 或 canonical target；
   当前 Mapping 不完整或不再正确，但 schema 明确显示所需 carrier 仍在当前 tables / join scope 内。

3. REGROUND_STRUCTURE
   新信息改变 entity、grain、relation、table scope 或 join path，或者 schema 明确显示所需 carrier
   位于当前 tables 之外。只有确有 table/join-scope 变化时才能选择。

4. TERMINAL
   当前没有足够、合法、可验证的信息选择一次有意义的 re-Grounding；重跑不能产生缺失的用户参数
   或 Official evidence；或者输入没有稳定的可验证 State。

约束：
- 不因为 State 本来可能有错就 restart；必须有本轮新信息造成的 invalidation。
- 不把 clarification 当 Official metadata。
- 不猜未知字段或表。无法证明需要新表时，不得升级为 REGROUND_STRUCTURE。
- 输出只是路由，不授权任何中间 State 对 Main 可见。
- 只返回 decision 和 reason 两个字段的裸 JSON。
""".strip()

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
MAPPING_REGROUNDING_PROMPT_SHA256 = hashlib.sha256(
    MAPPING_REGROUNDING_GROUNDING_PROMPT.encode("utf-8")
).hexdigest()
MAPPING_VALIDATION_CORRECTION_PROMPT_SHA256 = hashlib.sha256(
    MAPPING_VALIDATION_CORRECTION_PROMPT.encode("utf-8")
).hexdigest()
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
FINAL_REGROUNDING_GATE_PROMPT_SHA256 = hashlib.sha256(
    FINAL_REGROUNDING_GATE_PROMPT.encode("utf-8")
).hexdigest()
FINAL_REGROUNDING_GATE_FORM_SCHEMA = (
    FinalRegroundingGateResponse.model_json_schema()
)
FINAL_REGROUNDING_GATE_FORM_SCHEMA_SHA256 = hashlib.sha256(
    canonical_json(FINAL_REGROUNDING_GATE_FORM_SCHEMA).encode("utf-8")
).hexdigest()


def _prompt_for_call_kind(call_kind: GroundingCallKind) -> str:
    if call_kind == "final_gate":
        return FINAL_REGROUNDING_GATE_PROMPT
    if call_kind == "check" and resolved_literal_executable_carrier_enabled():
        from valibra_agent.sql_grounding.resolved_literal_proposal_shadow import (
            CHECK_RESOLVED_LITERAL_PROPOSAL_SHADOW_PROMPT,
        )

        return CHECK_RESOLVED_LITERAL_PROPOSAL_SHADOW_PROMPT
    return SQL_GROUNDING_STAGE_PROMPTS[call_kind]


def _prompt_for_input_payload(
    call_kind: GroundingCallKind,
    input_payload: Mapping[str, Any],
) -> str:
    if call_kind == "mapping" and "mapping_validation_correction" in input_payload:
        return MAPPING_VALIDATION_CORRECTION_PROMPT
    context = input_payload.get("regrounding_context")
    if (
        call_kind == "mapping"
        and isinstance(context, Mapping)
        and context.get("decision") in {"REGROUND_MAPPING", "REGROUND_STRUCTURE"}
    ):
        return MAPPING_REGROUNDING_GROUNDING_PROMPT
    return _prompt_for_call_kind(call_kind)


def _prompt_sha(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _form_schema_for_call_kind(call_kind: GroundingCallKind) -> dict[str, Any]:
    if call_kind == "final_gate":
        return FINAL_REGROUNDING_GATE_FORM_SCHEMA
    if call_kind == "check" and resolved_literal_executable_carrier_enabled():
        from valibra_agent.sql_grounding.resolved_literal_proposal_shadow import (
            CHECK_RESOLVED_LITERAL_PROPOSAL_SHADOW_FORM_SCHEMA,
        )

        return CHECK_RESOLVED_LITERAL_PROPOSAL_SHADOW_FORM_SCHEMA
    return SQL_GROUNDING_STAGE_FORM_SCHEMAS[call_kind]


def _prompt_sha_for_call_kind(call_kind: GroundingCallKind) -> str:
    if call_kind == "final_gate":
        return FINAL_REGROUNDING_GATE_PROMPT_SHA256
    if call_kind == "check" and resolved_literal_executable_carrier_enabled():
        from valibra_agent.sql_grounding.resolved_literal_proposal_shadow import (
            CHECK_RESOLVED_LITERAL_PROPOSAL_SHADOW_PROMPT_SHA256,
        )

        return CHECK_RESOLVED_LITERAL_PROPOSAL_SHADOW_PROMPT_SHA256
    return SQL_GROUNDING_STAGE_PROMPT_SHA256[call_kind]


def _form_sha_for_call_kind(call_kind: GroundingCallKind) -> str:
    if call_kind == "final_gate":
        return FINAL_REGROUNDING_GATE_FORM_SCHEMA_SHA256
    if call_kind == "check" and resolved_literal_executable_carrier_enabled():
        from valibra_agent.sql_grounding.resolved_literal_proposal_shadow import (
            CHECK_RESOLVED_LITERAL_PROPOSAL_SHADOW_FORM_SHA256,
        )

        return CHECK_RESOLVED_LITERAL_PROPOSAL_SHADOW_FORM_SHA256
    return SQL_GROUNDING_STAGE_FORM_SCHEMA_SHA256[call_kind]


def resolved_literal_executable_carrier_enabled() -> bool:
    """Return the explicit executable-carrier experiment switch."""

    return os.environ.get(
        "VALIBRA_RESOLVED_LITERAL_EXECUTABLE_CARRIER_R1"
    ) == "1"


def _max_tokens_for_call_kind(call_kind: GroundingCallKind) -> int:
    if call_kind == "final_gate":
        return FINAL_REGROUNDING_GATE_MAX_TOKENS
    return SQL_GROUNDING_STAGE_MAX_TOKENS[call_kind]


SQL_GROUNDING_PROMPT = canonical_json(
    {
        "stage_prompts": SQL_GROUNDING_STAGE_PROMPTS,
        "mapping_regrounding_prompt": MAPPING_REGROUNDING_GROUNDING_PROMPT,
    }
)
SQL_GROUNDING_PROMPT_SHA256 = hashlib.sha256(
    SQL_GROUNDING_PROMPT.encode("utf-8")
).hexdigest()
SQL_GROUNDING_FORM_SCHEMA = SQL_GROUNDING_STAGE_FORM_SCHEMAS
SQL_GROUNDING_FORM_SCHEMA_SHA256 = hashlib.sha256(
    canonical_json(SQL_GROUNDING_FORM_SCHEMA).encode("utf-8")
).hexdigest()
SQL_GROUNDING_CONFIGURATION = {
    "answered_clarification_routing": {
        "official_phase_unchanged": True,
        "routes": {
            "stay_check": ["check"],
            "restart_grounding": ["structure", "mapping", "knowledge", "check"],
            "terminal": [],
        },
        "state_external_overlay": True,
    },
    "check_phase_local_context": {
        "max_answered_clarifications": MAX_CHECK_PHASE_LOCAL_CLARIFICATIONS,
        "max_previous_official_calls": MAX_CHECK_PHASE_LOCAL_CALLS,
        "raw_tool_results": False,
    },
    "stage_form_schema_sha256": SQL_GROUNDING_STAGE_FORM_SCHEMA_SHA256,
    "stage_prompt_sha256": SQL_GROUNDING_STAGE_PROMPT_SHA256,
    "mapping_regrounding_prompt_sha256": MAPPING_REGROUNDING_PROMPT_SHA256,
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
        configured_max_tokens = _max_tokens_for_call_kind(request.call_kind)
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
            "prompt_sha256": _prompt_sha(request.prompt),
            "form_schema_sha256": _form_sha_for_call_kind(request.call_kind),
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
            or request.call_kind == "final_gate"
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


class KnowledgeRetirementAuditSidecar(ContractModel):
    """Non-authoritative digest of a stripped legacy Provider field."""

    value_shape: Literal["array", "non_array"]
    item_count: int = Field(ge=0, le=MAX_GROUNDING_RESPONSE_CHARS)
    value_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class GroundingUpdaterResult(ContractModel):
    response: GroundingLLMResponse | StageGroundingResponse
    call_kind: GroundingCallKind | None = None
    telemetry: GroundingLLMTelemetry
    transport_normalization: Literal["none", "single_json_fence"]
    knowledge_omission_mapping_ignored: tuple[str, ...] = ()
    knowledge_retirement_audit_sidecar: (
        KnowledgeRetirementAuditSidecar | None
    ) = None


class GroundingUpdaterError(RuntimeError):
    """A bounded failure; rejected Mapping content is private and in-memory only."""

    def __init__(
        self,
        reason: str,
        telemetry: GroundingLLMTelemetry,
        *,
        rejected_mapping_payload: str | None = None,
        validation_detail: str | None = None,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.telemetry = telemetry
        # These fields are an in-memory repair carrier only.  They are never
        # copied into ordinary telemetry or persisted by the service.
        self.rejected_mapping_payload = rejected_mapping_payload
        self.validation_detail = validation_detail


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
        mapping_validation_correction: Mapping[str, Any] | None = None,
    ) -> GroundingUpdaterResult:
        """Return a typed proposal without mutating Runtime or Observation."""

        request = _build_request(
            runtime,
            observation,
            original_query=original_query,
            follow_up_query=follow_up_query,
            grounding_input=grounding_input,
            mapping_validation_correction=mapping_validation_correction,
        )
        request_sha = hashlib.sha256(canonical_json(request).encode("utf-8")).hexdigest()
        request_prompt_sha = _prompt_sha(request.prompt)
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
                prompt_sha256=request_prompt_sha,
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
                prompt_sha256=request_prompt_sha,
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
                prompt_sha256=request_prompt_sha,
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
                prompt_sha256=request_prompt_sha,
                **provider_metadata,
            )
            raise GroundingUpdaterError("response_too_large", telemetry)
        try:
            payload, normalization = normalize_grounding_transport(
                client_response.content
            )
            knowledge_omission_mapping_ignored: tuple[str, ...] = ()
            knowledge_retirement_audit_sidecar = None
            if request.call_kind == "knowledge" and grounding_input is not None:
                payload, knowledge_retirement_audit_sidecar = (
                    _strip_knowledge_retirement_audit_sidecar(payload)
                )
                payload, knowledge_omission_mapping_ignored = (
                    _strip_knowledge_omission_owned_mappings(
                        payload,
                        grounding_input=grounding_input,
                    )
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
                prompt_sha256=request_prompt_sha,
                **provider_metadata,
            )
            raise GroundingUpdaterError(
                exc.reason,
                telemetry,
                rejected_mapping_payload=(
                    client_response.content
                    if request.call_kind == "mapping"
                    and exc.reason == "form_validation_failed"
                    and mapping_validation_correction is None
                    else None
                ),
                validation_detail=exc.detail,
            ) from exc

        telemetry = _telemetry(
            attempted=True,
            status="succeeded",
            usage=client_response.usage,
            latency_ms=_elapsed_ms(started),
            request_sha=effective_request_sha,
            response_sha=response_sha,
            call_kind=request.call_kind,
            prompt_sha256=request_prompt_sha,
            **provider_metadata,
        )
        return GroundingUpdaterResult(
            response=response,
            call_kind=request.call_kind,
            telemetry=telemetry,
            transport_normalization=normalization,
            knowledge_omission_mapping_ignored=(
                knowledge_omission_mapping_ignored
            ),
            knowledge_retirement_audit_sidecar=(
                knowledge_retirement_audit_sidecar
            ),
        )


class _StrictResponseError(ValueError):
    def __init__(self, reason: str, *, detail: str | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.detail = detail


def _build_request(
    runtime: GroundingRuntime,
    observation: SQLGroundingObservation,
    *,
    original_query: str,
    follow_up_query: str | None,
    grounding_input: Mapping[str, Any] | None,
    mapping_validation_correction: Mapping[str, Any] | None = None,
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
    if mapping_validation_correction is not None:
        if grounding_input is None or call_kind != "mapping":
            telemetry = _telemetry(
                attempted=False,
                status="rejected",
                error_type="grounding_bundle_invalid",
            )
            raise GroundingUpdaterError("grounding_bundle_invalid", telemetry)
        correction = dict(mapping_validation_correction)
        if (
            set(correction)
            != {"invalid_phrases", "rejected_mapping", "validation_error"}
            or not isinstance(correction["rejected_mapping"], dict)
            or not isinstance(correction["validation_error"], str)
            or not correction["validation_error"].strip()
            or not isinstance(correction["invalid_phrases"], list)
            or not correction["invalid_phrases"]
            or any(
                not isinstance(phrase, str) or not phrase.strip()
                for phrase in correction["invalid_phrases"]
            )
            or len(correction["invalid_phrases"])
            != len(set(correction["invalid_phrases"]))
        ):
            telemetry = _telemetry(
                attempted=False,
                status="rejected",
                error_type="grounding_bundle_invalid",
            )
            raise GroundingUpdaterError("grounding_bundle_invalid", telemetry)
        input_payload["mapping_validation_correction"] = correction
    input_json = canonical_json(input_payload)
    if len(input_json) > MAX_GROUNDING_REQUEST_CHARS:
        telemetry = _telemetry(
            attempted=False,
            status="rejected",
            error_type="request_too_large",
        )
        raise GroundingUpdaterError("request_too_large", telemetry)
    request = GroundingLLMRequest(
        prompt=_prompt_for_input_payload(call_kind, input_payload),
        input_json=input_json,
        response_schema=_form_schema_for_call_kind(call_kind),
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
    if call_kind in {"mapping", "knowledge", "check"}:
        raw_unresolved = payload.get("unresolved_mappings")
        if not isinstance(raw_unresolved, list):
            telemetry = _telemetry(
                attempted=False,
                status="rejected",
                error_type="grounding_bundle_invalid",
            )
            raise GroundingUpdaterError("grounding_bundle_invalid", telemetry)
        try:
            unresolved = tuple(
                UnresolvedMapping.model_validate(item)
                for item in raw_unresolved
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
        phrases = [item.phrase for item in unresolved]
        sources = [original_query]
        if follow_up_query is not None:
            sources.append(follow_up_query)
        mapped_phrases = {
            item.phrase
            for item in (runtime.grounding_state.column_mapping or ())
        }
        if (
            len(phrases) != len(set(phrases))
            or any(not any(phrase in source for source in sources) for phrase in phrases)
            or mapped_phrases & set(phrases)
        ):
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
    if "user_clarifications" in payload:
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
        if any(
            item.phase > observation.phase or item.answer is None
            for item in records
        ):
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
    if "regrounding_context" in payload:
        context = payload["regrounding_context"]
        if not isinstance(context, dict) or set(context) != {
            "decision",
            "check_gap",
            "answered_clarifications",
            "official_knowledge",
        }:
            telemetry = _telemetry(
                attempted=False,
                status="rejected",
                error_type="grounding_bundle_invalid",
            )
            raise GroundingUpdaterError("grounding_bundle_invalid", telemetry)
        decision = context.get("decision")
        gap = context.get("check_gap")
        if decision not in {"REGROUND_MAPPING", "REGROUND_STRUCTURE"} or (
            gap is not None
            and (
                not isinstance(gap, str)
                or not gap
                or gap != gap.strip()
                or len(gap) > 4_096
            )
        ):
            telemetry = _telemetry(
                attempted=False,
                status="rejected",
                error_type="grounding_bundle_invalid",
            )
            raise GroundingUpdaterError("grounding_bundle_invalid", telemetry)
        expected_clarifications = payload.get("user_clarifications", [])
        current_state = payload.get("current_state")
        expected_knowledge = (
            current_state.get("domain_knowledge")
            if isinstance(current_state, dict)
            else None
        )
        if (
            context.get("answered_clarifications") != expected_clarifications
            or context.get("official_knowledge") != expected_knowledge
        ):
            telemetry = _telemetry(
                attempted=False,
                status="rejected",
                error_type="grounding_bundle_invalid",
            )
            raise GroundingUpdaterError("grounding_bundle_invalid", telemetry)
    if call_kind == "final_gate":
        gate_context = payload.get("final_gate_context")
        if not isinstance(gate_context, dict) or set(gate_context) != {
            "answered_clarifications",
            "column_meanings",
            "final_check",
            "mapping_owner_transition",
            "previous_official_calls",
            "schema",
        }:
            telemetry = _telemetry(
                attempted=False,
                status="rejected",
                error_type="grounding_bundle_invalid",
            )
            raise GroundingUpdaterError("grounding_bundle_invalid", telemetry)
        if (
            not isinstance(gate_context["answered_clarifications"], list)
            or not isinstance(gate_context["previous_official_calls"], list)
            or not isinstance(gate_context["final_check"], dict)
            or not isinstance(gate_context["schema"], str)
            or not isinstance(gate_context["column_meanings"], dict)
        ):
            telemetry = _telemetry(
                attempted=False,
                status="rejected",
                error_type="grounding_bundle_invalid",
            )
            raise GroundingUpdaterError("grounding_bundle_invalid", telemetry)
        owner_transition = gate_context["mapping_owner_transition"]
        if not isinstance(owner_transition, dict) or set(owner_transition) != {
            "actionable_evidence_changed_since_mapping",
            "eligible_owner_transition_omissions",
            "mapping_is_only_owner",
            "unresolved_mappings",
        }:
            telemetry = _telemetry(
                attempted=False,
                status="rejected",
                error_type="grounding_bundle_invalid",
            )
            raise GroundingUpdaterError("grounding_bundle_invalid", telemetry)
        raw_owner_unresolved = owner_transition["unresolved_mappings"]
        raw_owner_eligible = owner_transition[
            "eligible_owner_transition_omissions"
        ]
        try:
            owner_unresolved = tuple(
                UnresolvedMapping.model_validate(item)
                for item in raw_owner_unresolved
            )
            owner_eligible = tuple(
                UnresolvedMapping.model_validate(item)
                for item in raw_owner_eligible
            )
        except (TypeError, ValidationError) as exc:
            telemetry = _telemetry(
                attempted=False,
                status="rejected",
                error_type="grounding_bundle_invalid",
            )
            raise GroundingUpdaterError(
                "grounding_bundle_invalid", telemetry
            ) from exc
        owner_unresolved_payloads = {
            canonical_json(item.model_dump(mode="json"))
            for item in owner_unresolved
        }
        owner_eligible_payloads = {
            canonical_json(item.model_dump(mode="json"))
            for item in owner_eligible
        }
        if (
            not isinstance(raw_owner_unresolved, list)
            or not isinstance(raw_owner_eligible, list)
            or owner_transition["mapping_is_only_owner"] is not True
            or not isinstance(
                owner_transition[
                    "actionable_evidence_changed_since_mapping"
                ],
                bool,
            )
            or len(owner_unresolved) != len(owner_unresolved_payloads)
            or len(owner_eligible) != len(owner_eligible_payloads)
            or not owner_eligible_payloads <= owner_unresolved_payloads
            or any(
                item.reason != "derived_rule_required"
                for item in owner_eligible
            )
            or (
                owner_transition[
                    "actionable_evidence_changed_since_mapping"
                ]
                and not owner_eligible
            )
        ):
            telemetry = _telemetry(
                attempted=False,
                status="rejected",
                error_type="grounding_bundle_invalid",
            )
            raise GroundingUpdaterError("grounding_bundle_invalid", telemetry)
    if call_kind == "check":
        previous_calls = payload.get("previous_official_calls")
        answered_clarifications = payload.get("answered_clarifications")
        cumulative_evidence = payload.get("check_evidence_context")
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
        if "check_evidence_context" in payload:
            if (
                not isinstance(cumulative_evidence, list)
                or len(cumulative_evidence)
                > MAX_P2_CHECK_CUMULATIVE_EVIDENCE_ENTRIES
                or len(canonical_json(cumulative_evidence))
                > MAX_P2_CHECK_CUMULATIVE_EVIDENCE_CHARS
            ):
                telemetry = _telemetry(
                    attempted=False,
                    status="rejected",
                    error_type="grounding_bundle_invalid",
                )
                raise GroundingUpdaterError("grounding_bundle_invalid", telemetry)
            cumulative_digests: list[str] = []
            for item in cumulative_evidence:
                if not isinstance(item, dict) or set(item) != {
                    "arguments",
                    "request_digest",
                    "result",
                    "result_sha256",
                    "source",
                    "tool_name",
                }:
                    telemetry = _telemetry(
                        attempted=False,
                        status="rejected",
                        error_type="grounding_bundle_invalid",
                    )
                    raise GroundingUpdaterError(
                        "grounding_bundle_invalid", telemetry
                    )
                tool_name = item.get("tool_name")
                arguments = item.get("arguments")
                request_digest = item.get("request_digest")
                result = item.get("result")
                result_sha256 = item.get("result_sha256")
                if (
                    tool_name != "get_column_meaning"
                    or not isinstance(arguments, dict)
                    or set(arguments) != {"table_name", "column_name"}
                    or not all(
                        isinstance(value, str)
                        and value
                        and value == value.strip()
                        and len(value) <= 256
                        for value in arguments.values()
                    )
                    or not isinstance(result, str)
                    or not result
                    or len(result) > MAX_P2_CHECK_CUMULATIVE_RESULT_CHARS
                    or item.get("source")
                    not in {"official_call", "p2_exact_p1_replay"}
                ):
                    telemetry = _telemetry(
                        attempted=False,
                        status="rejected",
                        error_type="grounding_bundle_invalid",
                    )
                    raise GroundingUpdaterError(
                        "grounding_bundle_invalid", telemetry
                    )
                expected_request_digest = hashlib.sha256(
                    f"{tool_name}:{canonical_json(arguments)}".encode("utf-8")
                ).hexdigest()
                expected_result_sha256 = hashlib.sha256(
                    canonical_json(result).encode("utf-8")
                ).hexdigest()
                if (
                    request_digest != expected_request_digest
                    or result_sha256 != expected_result_sha256
                ):
                    telemetry = _telemetry(
                        attempted=False,
                        status="rejected",
                        error_type="grounding_bundle_invalid",
                    )
                    raise GroundingUpdaterError(
                        "grounding_bundle_invalid", telemetry
                    )
                cumulative_digests.append(request_digest)
            if len(cumulative_digests) != len(set(cumulative_digests)):
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
    "mapping": frozenset({"column_meanings", "unresolved_mappings"}),
    "knowledge": frozenset(
        {
            "knowledge_definitions",
            "relevant_column_meanings",
            "unresolved_mappings",
        }
    ),
    "check": frozenset(
        {
            "answered_clarifications",
            "check_context",
            "previous_official_calls",
            "unresolved_mappings",
        }
    ),
    "final_gate": frozenset({"final_gate_context"}),
}

_CHECK_PHASE_LOCAL_CONTEXT_FIELDS = frozenset(
    {
        "answered_clarifications",
        "previous_official_calls",
        "unresolved_mappings",
    }
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
    common_variants: tuple[set[str], ...]
    if phase == 2:
        common.update({"follow_up", "user_clarifications"})
        common_variants = (common,)
    elif phase == 1:
        common_variants = (common, common | {"user_clarifications"})
    elif phase != 1:
        raise ValueError("Grounding phase must be 1 or 2")
    fields = set(grounding_input)
    stage_common_variants = tuple(
        variant
        for candidate_common in common_variants
        for variant in (
            candidate_common,
            candidate_common | {"regrounding_context"},
        )
    )
    matches: list[GroundingCallKind] = []
    for candidate_common in stage_common_variants:
        for call_kind, evidence_fields in _CALL_KIND_EVIDENCE_FIELDS.items():
            evidence_variants = [set(evidence_fields)]
            if (
                any(
                    fields == candidate_common | candidate_evidence
                    for candidate_evidence in evidence_variants
                )
                and (
                    "regrounding_context" not in candidate_common
                    or call_kind != "final_gate"
                )
            ):
                matches.append(call_kind)
    check_suffixes = tuple(
        context_fields | {latest_field}
        for context_fields in (
            _CHECK_PHASE_LOCAL_CONTEXT_FIELDS,
            _CHECK_PHASE_LOCAL_CONTEXT_FIELDS | {"check_evidence_context"},
        )
        for latest_field in ("check_context", "latest_tool", "latest_user_answer")
    )
    if "check" not in matches and any(
        fields == candidate_common | suffix
        for candidate_common in stage_common_variants
        for suffix in check_suffixes
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
            "final_gate": FinalRegroundingGateResponse,
            None: GroundingLLMResponse,
        }[call_kind]
        if call_kind == "check" and resolved_literal_executable_carrier_enabled():
            from valibra_agent.sql_grounding.resolved_literal_proposal_shadow import (
                GroundingCheckResolvedLiteralProposalShadowResponse,
            )

            model = GroundingCheckResolvedLiteralProposalShadowResponse
        return model.model_validate(raw)
    except ValidationError as exc:
        raise _StrictResponseError(
            "form_validation_failed",
            detail=json.dumps(
                exc.errors(include_url=False, include_input=False),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ),
        ) from exc


def _strip_knowledge_retirement_audit_sidecar(
    payload: str,
) -> tuple[str, KnowledgeRetirementAuditSidecar | None]:
    """Remove a legacy retirement field before authoritative Knowledge parse.

    P2_KNOWLEDGE_RETIREMENT_DECOUPLING_R1: executable retirement does not yet
    exist, so Provider retirement content cannot participate in core response
    validity or State materialization.  Retain only a bounded digest summary for
    audit; never retain or interpret the Provider's proposed authority.
    """

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
    if not isinstance(raw, dict) or "knowledge_retirement_proposals" not in raw:
        return payload, None
    retired_value = raw.pop("knowledge_retirement_proposals")
    serialized_value = canonical_json(retired_value)
    sidecar = KnowledgeRetirementAuditSidecar(
        value_shape="array" if isinstance(retired_value, list) else "non_array",
        item_count=len(retired_value) if isinstance(retired_value, list) else 0,
        value_sha256=hashlib.sha256(serialized_value.encode("utf-8")).hexdigest(),
    )
    return canonical_json(raw), sidecar


def _strip_knowledge_omission_owned_mappings(
    payload: str,
    *,
    grounding_input: Mapping[str, Any],
) -> tuple[str, tuple[str, ...]]:
    """Drop only Knowledge mappings for exact Mapping-owned omissions.

    This narrow transport projection runs before the typed Knowledge form so
    even an invalid ``targets=[]`` object cannot poison an otherwise valid
    Official-knowledge selection.  It never repairs or invents a target.
    """

    raw_unresolved = grounding_input.get("unresolved_mappings")
    if not isinstance(raw_unresolved, list) or not raw_unresolved:
        return payload, ()
    unresolved = tuple(
        UnresolvedMapping.model_validate(item) for item in raw_unresolved
    )
    unresolved_phrases = {item.phrase for item in unresolved}
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
        return payload, ()
    mappings = raw.get("column_mapping")
    if not isinstance(mappings, list):
        return payload, ()
    ignored = tuple(
        sorted(
            {
                item.get("phrase")
                for item in mappings
                if isinstance(item, dict)
                and isinstance(item.get("phrase"), str)
                and item.get("phrase") in unresolved_phrases
            }
        )
    )
    if not ignored:
        return payload, ()
    ignored_set = set(ignored)
    raw["column_mapping"] = [
        item
        for item in mappings
        if not (
            isinstance(item, dict)
            and item.get("phrase") in ignored_set
        )
    ]
    return canonical_json(raw), ignored


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
    try:
        input_payload = json.loads(request.input_json)
    except (TypeError, ValueError) as exc:
        raise SQLGroundingProviderError(
            "SQL Grounding input JSON invalid"
        ) from exc
    if (
        not isinstance(input_payload, dict)
        or request.prompt
        != _prompt_for_input_payload(request.call_kind, input_payload)
    ):
        raise SQLGroundingProviderError("SQL Grounding prompt mismatch")
    if request.response_schema != _form_schema_for_call_kind(request.call_kind):
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
        "max_tokens": _max_tokens_for_call_kind(request.call_kind),
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
        "prompt_sha256": _prompt_sha(request.prompt),
        "form_schema_sha256": _form_sha_for_call_kind(request.call_kind),
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
        (
            call_kind in SQL_GROUNDING_EXACT_EMPTY_RETRY_STAGES
            or call_kind == "final_gate"
        )
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
    prompt_sha256: str | None = None,
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
            prompt_sha256
            if prompt_sha256 is not None
            else (
                _prompt_sha_for_call_kind(call_kind)
                if call_kind is not None
                else SQL_GROUNDING_PROMPT_SHA256
            )
        ),
        form_schema_sha256=(
            _form_sha_for_call_kind(call_kind)
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
