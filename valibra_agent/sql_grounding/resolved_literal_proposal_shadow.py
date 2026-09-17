"""Research-only resolved-literal proposal and deterministic verifier.

``CHECK_RESOLVED_LITERAL_PROPOSAL_SHADOW_R1`` deliberately has no runtime
wiring.  It gives a stage-local Check replay one additional, non-authoritative
proposal field and verifies that proposal without mutating four-dimensional
Grounding State, control flow, or Main context.

The verifier is intentionally narrower than semantic inference.  A proposal
is accepted only when one exact mapped phrase and target are paired with the
current ``get_column_meaning`` result, the proposed value is an exact member
of that target's Official enum, and the phrase has exactly one mechanically
verifiable lexical match among those enum values.  Check proposes; runtime
provenance is generated only after these deterministic checks pass.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Annotated, Any, Literal, Mapping

import sqlglot
from pydantic import Field, field_validator, model_validator
from sqlglot import exp

from valibra_agent.sql_grounding.models import (
    ContractModel,
    GroundingCheckResponse,
    MAX_EXPRESSION_CHARS,
    MAX_PHRASE_CHARS,
    SQLGroundingState,
    canonical_json,
    canonicalize_field_expression,
    sql_grounding_state_sha256,
)
from valibra_agent.sql_grounding.updater import CHECK_GROUNDING_PROMPT


MAX_RESOLVED_LITERAL_CHARS = 128
MAX_RESOLVED_LITERAL_PROPOSALS = 1

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_POSSIBLE_VALUES_RE = re.compile(
    r"\bPossible values:\s*(.+?)(?:\.(?:\s|$)|$)",
    flags=re.IGNORECASE,
)
_NEGATIVE_TOKENS = frozenset(
    {
        "except",
        "exclude",
        "excluding",
        "neither",
        "no",
        "nor",
        "not",
        "without",
    }
)

_BASE_OUTPUT_CONTRACT = """只返回下面六个顶层字段：
status、clarification_route、missing_information、next_tool、column_mapping、domain_knowledge。"""
_SHADOW_OUTPUT_CONTRACT = """只返回下面七个顶层字段：
status、clarification_route、missing_information、next_tool、column_mapping、domain_knowledge、
resolved_literal_proposals。"""

_SHADOW_INSTRUCTION = """

Resolved literal proposal Shadow（只观测，不修改 State 或路由）：
- resolved_literal_proposals 必须始终返回数组；没有满足全部条件的 proposal 时返回 []；最多一项。
- 只有本轮 status=complete、latest_tool.name=get_column_meaning，且一个 exact
  current_state.column_mapping.phrase 已映射到一个 exact target 时，才可提出 proposal。
- proposal 只能包含 phrase、target、literal。phrase 与 target 必须逐字复制该 mapping；literal
  必须逐字复制 latest_tool 对这个 exact target 声明的一个 Possible values 枚举值。
- 只提议正向单值等号语义 target = literal。NOT、!=、IN、多值、范围、threshold、比较运算、
  AND/OR、formula、aggregation、grain 或任意 SQL expression 都返回 []。
- Query phrase 必须与 Official enum literal 存在直接、唯一、机械可验证的词形对应，例如
  fails/failed 或 highest/high。若零个或两个以上 enum literal 都能匹配，返回 []。
- 不得仅根据 serious/bad/breaks 等业务近义判断选择 High/Critical 等枚举；不得把 Check reasoning
  当作 authority。Official enum 只证明值存在，Query 的唯一词形对应才授权本 Shadow proposal。
- 该字段只是 Provider proposal；不得改变原有六个字段的判断，不得填写 authority、operator、
  source digest、phase 或 revision。
""".strip()


def build_resolved_literal_proposal_shadow_prompt(
    base_prompt: str = CHECK_GROUNDING_PROMPT,
) -> str:
    """Derive the isolated Shadow prompt while leaving the base prompt intact."""

    if base_prompt.count(_BASE_OUTPUT_CONTRACT) != 1:
        raise ValueError("Check output contract anchor is not unique")
    shadow = base_prompt.replace(
        _BASE_OUTPUT_CONTRACT,
        _SHADOW_OUTPUT_CONTRACT,
        1,
    )
    return f"{shadow}\n\n{_SHADOW_INSTRUCTION}"


CHECK_RESOLVED_LITERAL_PROPOSAL_SHADOW_PROMPT = (
    build_resolved_literal_proposal_shadow_prompt()
)
CHECK_RESOLVED_LITERAL_PROPOSAL_SHADOW_PROMPT_SHA256 = hashlib.sha256(
    CHECK_RESOLVED_LITERAL_PROPOSAL_SHADOW_PROMPT.encode("utf-8")
).hexdigest()


class ResolvedLiteralProposal(ContractModel):
    """Non-authoritative Check proposal; no provenance may be self-asserted."""

    phrase: Annotated[str, Field(min_length=1, max_length=MAX_PHRASE_CHARS)]
    target: Annotated[str, Field(min_length=1, max_length=MAX_EXPRESSION_CHARS)]
    literal: Annotated[
        str,
        Field(min_length=1, max_length=MAX_RESOLVED_LITERAL_CHARS),
    ]

    @field_validator("phrase", "literal")
    @classmethod
    def validate_bounded_text(cls, value: str) -> str:
        if value != value.strip() or any(ord(character) < 32 for character in value):
            raise ValueError("proposal text must have no outer whitespace or controls")
        return value

    @field_validator("target")
    @classmethod
    def validate_target(cls, value: str) -> str:
        canonicalize_field_expression(value)
        return value


class GroundingCheckResolvedLiteralProposalShadowResponse(GroundingCheckResponse):
    """The current Check form plus one required audit-only proposal array."""

    resolved_literal_proposals: Annotated[
        tuple[ResolvedLiteralProposal, ...],
        Field(max_length=MAX_RESOLVED_LITERAL_PROPOSALS),
    ]

    @model_validator(mode="after")
    def validate_shadow_proposal_mode(
        self,
    ) -> "GroundingCheckResolvedLiteralProposalShadowResponse":
        if self.resolved_literal_proposals and self.status != "complete":
            raise ValueError("only complete Check may emit a resolved literal proposal")
        return self


CHECK_RESOLVED_LITERAL_PROPOSAL_SHADOW_FORM_SCHEMA = (
    GroundingCheckResolvedLiteralProposalShadowResponse.model_json_schema()
)
CHECK_RESOLVED_LITERAL_PROPOSAL_SHADOW_FORM_SHA256 = hashlib.sha256(
    canonical_json(
        CHECK_RESOLVED_LITERAL_PROPOSAL_SHADOW_FORM_SCHEMA
    ).encode("utf-8")
).hexdigest()


class ResolvedLiteralShadowCarrier(ContractModel):
    """Runtime-authored provenance for one verified Shadow proposal."""

    task_id: Annotated[str, Field(min_length=1, max_length=256)]
    phase: Literal[1, 2]
    grounding_revision: Annotated[int, Field(ge=0)]
    state_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    phrase: Annotated[str, Field(min_length=1, max_length=MAX_PHRASE_CHARS)]
    target: Annotated[str, Field(min_length=1, max_length=MAX_EXPRESSION_CHARS)]
    literal: Annotated[
        str,
        Field(min_length=1, max_length=MAX_RESOLVED_LITERAL_CHARS),
    ]
    operator: Literal["EXACT_EQUALITY"] = "EXACT_EQUALITY"
    authority: Literal[
        "query_lexical_literal+official_column_meaning"
    ] = "query_lexical_literal+official_column_meaning"
    source_request_digest: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    source_result_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class ResolvedLiteralShadowValidation(ContractModel):
    """One bounded, auditable verdict for the complete Shadow response."""

    verdict: Literal["VALID", "OMITTED", "REJECTED"]
    reason: Annotated[str, Field(min_length=1, max_length=128)]
    matching_enum_literals: Annotated[
        tuple[Annotated[str, Field(min_length=1, max_length=MAX_RESOLVED_LITERAL_CHARS)], ...],
        Field(max_length=16),
    ] = ()
    carrier: ResolvedLiteralShadowCarrier | None = None

    @model_validator(mode="after")
    def validate_verdict_shape(self) -> "ResolvedLiteralShadowValidation":
        if (self.verdict == "VALID") != (self.carrier is not None):
            raise ValueError("only VALID verdict may contain a Shadow carrier")
        return self


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _token_roots(token: str) -> frozenset[str]:
    word = token.casefold()
    roots = {word}
    # Deliberately finite English morphology for the two approved authority
    # shapes.  It does not use synonyms or semantic similarity.
    for suffix, minimum in (
        ("est", 6),
        ("ing", 6),
        ("ed", 5),
        ("es", 5),
        ("er", 5),
        ("s", 4),
    ):
        if len(word) >= minimum and word.endswith(suffix):
            root = word[: -len(suffix)]
            if len(root) >= 3:
                roots.add(root)
    return frozenset(roots)


def _lexical_tokens(value: str) -> tuple[frozenset[str], ...]:
    return tuple(_token_roots(token) for token in _TOKEN_RE.findall(value))


def _literal_matches_phrase(phrase: str, literal: str) -> bool:
    phrase_tokens = _lexical_tokens(phrase)
    literal_tokens = _lexical_tokens(literal)
    if not phrase_tokens or not literal_tokens:
        return False
    if len(literal_tokens) == 1:
        return any(literal_tokens[0] & candidate for candidate in phrase_tokens)
    width = len(literal_tokens)
    return any(
        all(literal_tokens[offset] & window[offset] for offset in range(width))
        for start in range(len(phrase_tokens) - width + 1)
        for window in (phrase_tokens[start : start + width],)
    )


def _json_path_keys(path: exp.Expression) -> tuple[str, ...]:
    if not isinstance(path, exp.JSONPath):
        raise ValueError("target JSON path is not canonical")
    result: list[str] = []
    for item in path.expressions:
        if isinstance(item, exp.JSONPathRoot):
            continue
        if not isinstance(item, exp.JSONPathKey) or not isinstance(item.this, str):
            raise ValueError("target JSON path contains a non-key component")
        result.append(item.this)
    if not result:
        raise ValueError("target JSON path is empty")
    return tuple(result)


def _target_reference(
    expression: exp.Expression,
) -> tuple[str, str, tuple[str, ...]]:
    if isinstance(expression, exp.Paren):
        return _target_reference(expression.this)
    if isinstance(expression, (exp.JSONExtract, exp.JSONExtractScalar)):
        table, column, prefix = _target_reference(expression.this)
        return table, column, prefix + _json_path_keys(expression.expression)
    if isinstance(expression, exp.Column):
        if not expression.table or not expression.name:
            raise ValueError("target must reference one fully qualified column")
        return expression.table, expression.name, ()
    raise ValueError("target is not one direct column or JSON leaf")


def _official_payload(raw_result: Any) -> tuple[Mapping[str, Any] | None, str | None]:
    if isinstance(raw_result, Mapping):
        return raw_result, None
    if not isinstance(raw_result, str) or not raw_result:
        raise ValueError("Official column meaning result is absent")
    try:
        parsed = json.loads(raw_result)
    except json.JSONDecodeError:
        return None, raw_result
    if not isinstance(parsed, Mapping):
        raise ValueError("Official column meaning JSON is not an object")
    return parsed, None


def _target_description(
    raw_result: Any,
    *,
    path: tuple[str, ...],
) -> str:
    payload, plain = _official_payload(raw_result)
    if not path:
        if plain is not None:
            return plain
        meaning = payload.get("column_meaning") if payload is not None else None
        if isinstance(meaning, str) and meaning:
            return meaning
        raise ValueError("Official direct-column meaning is unavailable")
    if payload is None:
        raise ValueError("Official JSON field meanings are unavailable")
    node: Any = payload.get("fields_meaning")
    for key in path:
        if not isinstance(node, Mapping) or key not in node:
            raise ValueError("target JSON path is absent from Official evidence")
        node = node[key]
    if not isinstance(node, str) or not node:
        raise ValueError("target JSON leaf does not have a scalar meaning")
    return node


def _possible_values(description: str) -> tuple[str, ...]:
    match = _POSSIBLE_VALUES_RE.search(description)
    if match is None:
        raise ValueError("target meaning has no bounded Possible values enum")
    values = tuple(item.strip() for item in match.group(1).split(","))
    if (
        not values
        or any(
            not item
            or len(item) > MAX_RESOLVED_LITERAL_CHARS
            or any(ord(character) < 32 for character in item)
            for item in values
        )
        or len(values) != len(set(values))
        or len(values) > 16
    ):
        raise ValueError("Official Possible values enum is not bounded and unique")
    return values


def _rejected(
    reason: str,
    *,
    matching: tuple[str, ...] = (),
) -> ResolvedLiteralShadowValidation:
    return ResolvedLiteralShadowValidation(
        verdict="REJECTED",
        reason=reason,
        matching_enum_literals=matching,
    )


def validate_resolved_literal_proposal_shadow(
    response: GroundingCheckResolvedLiteralProposalShadowResponse,
    *,
    grounding_input: Mapping[str, Any],
    task_id: str,
    phase: Literal[1, 2],
    grounding_revision: int,
) -> ResolvedLiteralShadowValidation:
    """Validate one Shadow response without changing any supplied object."""

    proposals = response.resolved_literal_proposals
    if not proposals:
        return ResolvedLiteralShadowValidation(
            verdict="OMITTED",
            reason="no_resolved_literal_proposal",
        )
    proposal = proposals[0]
    if response.status != "complete":
        return _rejected("check_not_complete")

    try:
        state = SQLGroundingState.model_validate(grounding_input.get("current_state"))
    except Exception:
        return _rejected("current_state_invalid")
    mappings = tuple(
        item
        for item in state.column_mapping or ()
        if item.phrase == proposal.phrase
    )
    if len(mappings) != 1:
        return _rejected("phrase_not_exact_current_mapping")
    mapping = mappings[0]
    if len(mapping.targets) != 1 or proposal.target != mapping.targets[0]:
        return _rejected("target_not_unique_exact_current_mapping")

    query_parts = [grounding_input.get("query")]
    if phase == 2:
        query_parts.append(grounding_input.get("follow_up"))
    if not any(
        isinstance(item, str) and proposal.phrase in item
        for item in query_parts
    ):
        return _rejected("phrase_not_verbatim_query_span")
    phrase_words = {token.casefold() for token in _TOKEN_RE.findall(proposal.phrase)}
    if phrase_words & _NEGATIVE_TOKENS:
        return _rejected("negative_or_exclusion_phrase_not_supported")

    latest_tool = grounding_input.get("latest_tool")
    if not isinstance(latest_tool, Mapping) or latest_tool.get("name") != "get_column_meaning":
        return _rejected("latest_tool_not_get_column_meaning")
    arguments = latest_tool.get("arguments")
    if not isinstance(arguments, Mapping) or set(arguments) != {
        "table_name",
        "column_name",
    }:
        return _rejected("official_tool_arguments_invalid")
    try:
        expression = sqlglot.parse_one(proposal.target, read="postgres")
        table, column, path = _target_reference(expression)
    except Exception:
        return _rejected("target_reference_invalid")
    if (
        arguments.get("table_name") != table
        or arguments.get("column_name") != column
    ):
        return _rejected("official_evidence_target_mismatch")

    raw_result = latest_tool.get("result")
    try:
        description = _target_description(raw_result, path=path)
        enum_values = _possible_values(description)
    except ValueError as exc:
        return _rejected(str(exc).replace(" ", "_").casefold()[:128])
    if proposal.literal not in enum_values:
        return _rejected("literal_not_exact_official_enum_member")
    matching = tuple(
        value
        for value in enum_values
        if _literal_matches_phrase(proposal.phrase, value)
    )
    if len(matching) != 1:
        return _rejected("query_enum_lexical_match_not_unique", matching=matching)
    if matching[0] != proposal.literal:
        return _rejected("proposal_not_unique_lexical_enum_match", matching=matching)

    request_identity = {
        "arguments": dict(arguments),
        "name": "get_column_meaning",
    }
    raw_result_text = (
        raw_result
        if isinstance(raw_result, str)
        else canonical_json(raw_result)
    )
    carrier = ResolvedLiteralShadowCarrier(
        task_id=task_id,
        phase=phase,
        grounding_revision=grounding_revision,
        state_sha256=sql_grounding_state_sha256(state),
        phrase=proposal.phrase,
        target=proposal.target,
        literal=proposal.literal,
        source_request_digest=_sha256_text(canonical_json(request_identity)),
        source_result_sha256=_sha256_text(raw_result_text),
    )
    return ResolvedLiteralShadowValidation(
        verdict="VALID",
        reason="unique_query_lexeme_and_exact_official_enum",
        matching_enum_literals=matching,
        carrier=carrier,
    )

