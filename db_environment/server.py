"""DB Environment Service (Port 6002). SQL execution, submission, schema/knowledge."""

import asyncio
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException

from db_environment.fixture_compat import (
    environment_sql,
    run_custom_test_override,
)
from shared.config import settings
from shared.db_utils import (
    _get_or_init_pool, close_pool, execute_queries,
    test_case_default,
    ex_base, remove_distinct, remove_comments, remove_round,
    clone_database, reset_task_db, drop_task_db, task_database_names,
    split_sql_statements,
)
from shared.models import (
    ExecuteSQLRequest, ExecuteSQLResponse, InitTaskRequest,
    SchemaRequest, ColumnMeaningRequest, KnowledgeRequest,
    SubmitSQLRequest, SubmitSQLResponse,
)

logger = logging.getLogger(__name__)
app = FastAPI(title="BIRD-Interact DB Environment", version="1.0.0")

MAX_RESULT_LENGTH = 500
KNOWLEDGE_VISIBLE_FIELDS = ["id", "knowledge", "description", "definition"]

# Some source KB files contain LaTeX commands with a single backslash even
# though JSON requires the backslash itself to be escaped. A subset of those
# commands collide with otherwise-valid JSON escapes (for example ``\text``
# becomes a tab plus ``ext`` and ``\frac`` becomes a form-feed plus ``rac``),
# so json.loads() succeeds after silently changing the Official definition.
# Keep this list limited to command spellings observed in the shipped KBs.
_SOURCE_KB_LATEX_COMMANDS = frozenset(
    {"bar", "begin", "beta", "frac", "text", "times"}
)

_task_data: Dict[str, Dict[str, Any]] = {}
_schema_cache: Dict[str, str] = {}
_column_meanings_cache: Dict[str, Dict] = {}
_external_knowledge_cache: Dict[str, Dict] = {}
_submit_attempts: Dict[str, Dict[int, int]] = {}
_successful_phase1_sql: Dict[str, str] = {}


class DatabaseStateError(RuntimeError):
    """Infrastructure failure while preparing/restoring an evaluation database."""


def _preserve_source_kb_latex_escapes(raw_line: str) -> str:
    """Escape known single-backslash LaTeX commands before JSON decoding.

    Already-correct ``\\command`` spellings and ordinary JSON escapes such as
    ``\n`` remain byte-for-byte unchanged. This is a source-carrier repair,
    not a semantic rewrite of a Knowledge definition.
    """

    result: list[str] = []
    index = 0
    while index < len(raw_line):
        character = raw_line[index]
        if character != "\\":
            result.append(character)
            index += 1
            continue

        run_end = index
        while run_end < len(raw_line) and raw_line[run_end] == "\\":
            run_end += 1
        slash_count = run_end - index
        result.append("\\" * slash_count)

        # An odd run leaves one source-level escape. Preserve it when it is a
        # known LaTeX command; an even run is already JSON-safe.
        if slash_count % 2 == 1:
            command_end = run_end
            while command_end < len(raw_line) and raw_line[command_end].isalpha():
                command_end += 1
            command = raw_line[run_end:command_end]
            if command in _SOURCE_KB_LATEX_COMMANDS:
                result.append("\\")
        index = run_end
    return "".join(result)


def _decode_source_knowledge_entry(raw_line: str) -> Dict[str, Any]:
    """Decode one source KB record without allowing lexical escape loss."""

    value = json.loads(_preserve_source_kb_latex_escapes(raw_line))
    if not isinstance(value, dict):
        raise ValueError("knowledge record must be an object")
    knowledge_id = value.get("id")
    knowledge_name = value.get("knowledge")
    definition = value.get("definition")
    if isinstance(knowledge_id, bool) or not isinstance(knowledge_id, int):
        raise ValueError("knowledge record must have an integer id")
    if not isinstance(knowledge_name, str) or not knowledge_name.strip():
        raise ValueError("knowledge record must have a non-empty name")
    if (
        not isinstance(definition, str)
        or not definition
        or definition != definition.strip()
    ):
        raise ValueError("knowledge record must have a canonical definition")
    return value


def _load_knowledge_catalog(path: str) -> Dict[str, Dict[str, Any]]:
    """Load valid KB entries while quarantining a malformed record locally."""

    catalog: Dict[str, Dict[str, Any]] = {}
    with open(path, encoding="utf-8") as knowledge_file:
        for line_number, raw_line in enumerate(knowledge_file, start=1):
            if not raw_line.strip():
                continue
            try:
                entry = _decode_source_knowledge_entry(raw_line.strip())
            except (json.JSONDecodeError, ValueError, TypeError) as exc:
                logger.error(
                    "Knowledge entry quarantined at %s:%s: %s: %s",
                    path,
                    line_number,
                    type(exc).__name__,
                    exc,
                )
                continue
            catalog[entry["knowledge"]] = entry
    return catalog


def _normalise_sql_sequence(value: Any) -> List[str]:
    if value is None:
        return []
    values = value if isinstance(value, list) else [value]
    return [item for item in values if isinstance(item, str) and item.strip()]


def _phase_maintenance_sql(statements: List[str]) -> List[str]:
    """Return P1 maintenance operations whose effects DB cloning omits.

    PostgreSQL CREATE DATABASE ... TEMPLATE copies relational state but not
    cumulative statistics such as pg_stat_all_tables.last_analyze.  Replaying
    ANALYZE after a P1 snapshot restore preserves the official single-database
    phase-transition semantics without repeating P1 writes.
    """
    return [
        statement
        for statement in _normalise_sql_sequence(statements)
        if re.match(r"^\s*ANALYZE\b", statement, flags=re.IGNORECASE)
    ]


def _execute_required_queries(db_name: str, queries: Any, *, label: str) -> None:
    """Execute required environment SQL and fail the task on any DB error."""
    statements = _normalise_sql_sequence(queries)
    if not statements:
        return
    pool = _get_or_init_pool(db_name)
    conn = pool.getconn()
    try:
        _, error, timed_out, _ = execute_queries(statements, db_name, conn)
        if timed_out:
            raise DatabaseStateError(f"{label} timed out")
        if error:
            raise DatabaseStateError(f"{label} failed: {error}")
    finally:
        pool.putconn(conn)


def _drop_databases(database_names: List[str], *, suppress_errors: bool) -> None:
    errors = []
    for database_name in dict.fromkeys(name for name in database_names if name):
        try:
            drop_task_db(database_name)
        except Exception as exc:
            errors.append(f"{database_name}: {type(exc).__name__}: {exc}")
    if errors and not suppress_errors:
        raise DatabaseStateError("Database cleanup failed: " + "; ".join(errors))


def _initialise_task_databases(task_id: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create the task DB and its canonical post-preprocess reset point."""
    base_db = task_data["selected_database"]
    base_template = f"{base_db}_template"
    names = task_database_names(base_db, task_id)
    owned_names = [names["phase1"], names["initial"], names["task"]]

    try:
        # Deterministic names make stale DBs recoverable after a process crash.
        _drop_databases(owned_names, suppress_errors=False)
        clone_database(names["task"], base_template)

        preprocess_sql = _normalise_sql_sequence(task_data.get("preprocess_sql"))
        initial_template = base_template
        owns_initial_snapshot = False
        if preprocess_sql:
            _execute_required_queries(
                names["task"], preprocess_sql, label="preprocess_sql"
            )
            close_pool(names["task"])
            clone_database(names["initial"], names["task"])
            initial_template = names["initial"]
            owns_initial_snapshot = True

        return {
            "_task_db": names["task"],
            "_initial_snapshot_db": initial_template,
            "_initial_snapshot_owned": owns_initial_snapshot,
            "_physical_db_names": names,
            "_preprocess_statement_count": len(preprocess_sql),
        }
    except Exception:
        _drop_databases(owned_names, suppress_errors=True)
        raise


def _create_phase1_snapshot(td: Dict[str, Any], pred_sqls: List[str]) -> str:
    """Build the phase-2 reset point from canonical P1 starting state."""
    task_db = td["_task_db"]
    initial_snapshot = td["_initial_snapshot_db"]
    phase1_snapshot = td["_physical_db_names"]["phase1"]

    reset_task_db(task_db, initial_snapshot)
    _execute_required_queries(task_db, pred_sqls, label="accepted Phase 1 SQL")
    td["_phase1_maintenance_sql"] = _phase_maintenance_sql(pred_sqls)
    cleanup_sql = td.get("clean_up_sqls", td.get("clean_up_sql"))
    cleanup_sql = environment_sql(
        td.get("instance_id", ""), "clean_up_sqls", cleanup_sql
    )
    _execute_required_queries(task_db, cleanup_sql, label="clean_up_sqls")
    close_pool(task_db)
    clone_database(phase1_snapshot, task_db)
    return phase1_snapshot


def _load_db_data(db_name: str):
    if db_name in _schema_cache:
        return
    db_folder = os.path.join(settings.db_data_path, db_name)
    # Schema
    try:
        with open(os.path.join(db_folder, f"{db_name}_schema.txt")) as f:
            _schema_cache[db_name] = f.read()
    except Exception as e:
        logger.error(f"Schema load failed for {db_name}: {e}")
        _schema_cache[db_name] = "Schema not available"
    # Column meanings
    try:
        with open(os.path.join(db_folder, f"{db_name}_column_meaning_base.json")) as f:
            raw = json.load(f)
        _column_meanings_cache[db_name] = {k.lower(): v for k, v in raw.items()}
    except Exception as e:
        logger.error(f"Column meanings load failed for {db_name}: {e}")
        _column_meanings_cache[db_name] = {}
    # Knowledge
    try:
        _external_knowledge_cache[db_name] = _load_knowledge_catalog(
            os.path.join(db_folder, f"{db_name}_kb.jsonl")
        )
    except Exception as e:
        logger.error(f"Knowledge load failed for {db_name}: {e}")
        _external_knowledge_cache[db_name] = {}


def _filter_knowledge(db_name: str, record: Dict) -> Dict:
    full_kb = _external_knowledge_cache.get(db_name, {})
    if not full_kb: return {}
    agent_kb = full_kb.copy()
    deleted_ids = set()
    for amb in record.get("knowledge_ambiguity", []):
        dk = amb.get("deleted_knowledge")
        if dk is not None: deleted_ids.add(dk)
    if deleted_ids:
        to_remove = [k for k, v in agent_kb.items() if v.get("id") in deleted_ids]
        for k in to_remove: del agent_kb[k]
    return agent_kb


def _format_result(result, cursor_desc=None) -> str:
    if result is None: return "Query executed successfully."
    if not isinstance(result, list): return str(result)
    if not result: return "Query executed, empty result set."
    lines = []
    if cursor_desc:
        cols = [desc[0] for desc in cursor_desc]
        lines.append(" | ".join(cols))
        lines.append("-" * min(len(lines[0]), 200))
    for row in result[:100]:
        cells = [str(c)[:100] for c in row]
        lines.append(" | ".join(cells))
    text = "\n".join(lines)
    words = text.split()
    if len(words) > MAX_RESULT_LENGTH:
        text = " ".join(words[:MAX_RESULT_LENGTH]) + "..."
    return text


@app.post("/init_task")
async def init_task(req: InitTaskRequest):
    task_data = dict(req.task_data)
    db_name = task_data["selected_database"]
    _load_db_data(db_name)
    # Never retain a stale in-memory reference if a same-ID reinitialization
    # fails after its deterministic physical databases have been replaced.
    _task_data.pop(req.task_id, None)
    _submit_attempts.pop(req.task_id, None)
    _successful_phase1_sql.pop(req.task_id, None)
    try:
        db_state = await asyncio.to_thread(
            _initialise_task_databases, req.task_id, task_data
        )
    except Exception as exc:
        logger.error("Task database initialization failed for %s", req.task_id, exc_info=True)
        raise HTTPException(
            status_code=503,
            detail=f"Task database initialization failed: {type(exc).__name__}: {exc}",
        ) from exc

    task_data.update(db_state)
    _task_data[req.task_id] = task_data
    _submit_attempts[req.task_id] = {1: 0, 2: 0}
    logger.info(
        "Initialized task %s: logical_db=%s physical_db=%s preprocess=%d",
        req.task_id,
        db_name,
        task_data["_task_db"],
        task_data["_preprocess_statement_count"],
    )
    return {
        "status": "ok",
        "task_id": req.task_id,
        "database_state": {
            "task_db": task_data["_task_db"],
            "initial_snapshot_db": task_data["_initial_snapshot_db"],
            "preprocess_statement_count": task_data["_preprocess_statement_count"],
        },
    }


def _execute_sql_sync(task_db: str, sql: str) -> ExecuteSQLResponse:
    """Blocking SQL execution — runs in thread pool."""
    try:
        pool = _get_or_init_pool(task_db)
        conn = pool.getconn()
        try:
            # Reset connection if it's in a bad state
            if conn.closed:
                pool.putconn(conn, close=True)
                conn = pool.getconn()
            try:
                conn.reset()
            except Exception as reset_err:
                logger.warning(f"conn.reset() failed for {task_db}: {reset_err}, getting fresh conn")
                try:
                    pool.putconn(conn, close=True)
                except Exception:
                    pass
                conn = pool.getconn()
            result, err, timeout, desc = execute_queries(sql, task_db, conn)
            if err:
                return ExecuteSQLResponse(result="", success=False, error=f"SQL error: {err}")
            if timeout:
                return ExecuteSQLResponse(result="", success=False, error="SQL execution timed out")
            formatted = _format_result(result, desc)
            return ExecuteSQLResponse(result=formatted, success=True)
        finally:
            try:
                pool.putconn(conn)
            except Exception:
                pass
    except Exception as e:
        logger.error(f"execute_sql error for {task_db}: {type(e).__name__}: {e}")
        error_msg = str(e) or f"{type(e).__name__}: {repr(e)}"
        return ExecuteSQLResponse(result="", success=False, error=error_msg)


@app.post("/execute", response_model=ExecuteSQLResponse)
async def execute_sql_endpoint(req: ExecuteSQLRequest):
    td = _task_data.get(req.task_id)
    if not td: raise HTTPException(404, f"Task {req.task_id} not initialized")
    task_db = td.get("_task_db", td["selected_database"])
    # SELECT-only: prevent agent from modifying DB state (strip comments first)
    sql_cleaned = re.sub(r'--.*$', '', req.sql, flags=re.MULTILINE)
    sql_cleaned = re.sub(r'/\*.*?\*/', '', sql_cleaned, flags=re.DOTALL)
    sql_upper = sql_cleaned.strip().upper()
    if not sql_upper.startswith(("SELECT", "WITH", "EXPLAIN")):
        return ExecuteSQLResponse(result="", success=False, error="Only SELECT queries allowed in execute_sql")
    return await asyncio.to_thread(_execute_sql_sync, task_db, req.sql)


def _submit_sql_sync(req_task_id, req_sql, td, _submit_attempts, _successful_phase1_sql) -> SubmitSQLResponse:
    """Blocking submit logic — runs in thread pool."""
    task_db = td["_task_db"]
    current_phase = td.get("_current_phase", 1)

    if req_task_id not in _submit_attempts:
        _submit_attempts[req_task_id] = {1: 0, 2: 0}
    _submit_attempts[req_task_id][current_phase] = _submit_attempts[req_task_id].get(current_phase, 0) + 1
    is_first_try = _submit_attempts[req_task_id][current_phase] == 1

    interact_mode = td.get("_interact_mode", "a-interact")
    phase_rewards_first = {1: 0.7, 2: 0.3}
    phase_rewards_debug = {1: 0.5, 2: 0.2}

    try:
        # Reset task DB for clean evaluation
        if current_phase == 1:
            template = td["_initial_snapshot_db"]
        else:
            # Phase 2: reset from Phase 1 snapshot (has Phase 1 state applied)
            template = td.get("_snapshot_db")
            if not template:
                raise DatabaseStateError("Phase 1 snapshot is missing before Phase 2")
        reset_task_db(task_db, template)
        if current_phase == 2:
            _execute_required_queries(
                task_db,
                td.get("_phase1_maintenance_sql"),
                label="Phase 1 maintenance replay",
            )

        pool = _get_or_init_pool(task_db)
        conn = pool.getconn()
        try:
            if current_phase == 2 and td.get("follow_up"):
                fu = td["follow_up"]
                sol_sqls = fu.get("sol_sql", [])
                test_cases = fu.get("test_cases", [])
                conditions = fu.get("conditions", {})
                category = fu.get("category", "Query")
            else:
                sol_sqls = td.get("sol_sql", [])
                test_cases = td.get("test_cases", [])
                conditions = td.get("conditions", {})
                category = td.get("category", "Query")

            if isinstance(sol_sqls, str): sol_sqls = [sol_sqls]
            pred_sqls = (
                split_sql_statements(req_sql)
                if isinstance(req_sql, str)
                else _normalise_sql_sequence(req_sql)
            )

            passed = False
            message = "Test case execution failed."

            if sol_sqls:
                # Execute pred SQL (also serves as executability check)
                pred_query_result, pred_err, pred_to, _ = execute_queries(pred_sqls, task_db, conn)
                if pred_err:
                    message = f"[exec_err_flg] Error executing submitted SQL: {pred_err}"
                elif pred_to:
                    message = "[exec_err_flg] Submitted SQL execution timed out"
                elif category == "Query" or not test_cases:
                    try:
                        test_case_default(pred_sqls, sol_sqls, task_db, conn, conditions)
                        passed = True
                        message = "SQL passed test case."
                    except AssertionError:
                        message = "Your SQL is not correct."
                    except Exception:
                        message = "Your SQL is not correct."
                else:
                    # Compat wrapper: custom test cases expect 3-value return (result, error, timeout)
                    def _execute_queries_compat(queries, db_name, conn=None):
                        result, error, timeout, _ = execute_queries(queries, db_name, conn)
                        return result, error, timeout

                    exec_globals = {
                        "execute_queries": _execute_queries_compat, "ex_base": ex_base,
                        "remove_distinct": remove_distinct, "remove_comments": remove_comments,
                        "remove_round": remove_round,
                        "pred_query_result": pred_query_result,
                    }
                    all_passed = True
                    for i, tc_code in enumerate(test_cases):
                        if not isinstance(tc_code, str): continue
                        try:
                            if run_custom_test_override(
                                task_id=req_task_id,
                                phase=current_phase,
                                test_index=i,
                                pred_sqls=pred_sqls,
                                db_name=task_db,
                                conn=conn,
                                execute_queries=_execute_queries_compat,
                            ):
                                continue
                            exec_locals = {}
                            exec(tc_code, exec_globals, exec_locals)
                            tc_func = exec_locals.get("test_case")
                            if tc_func and callable(tc_func):
                                tc_func(pred_sqls, sol_sqls, task_db, conn)
                        except AssertionError:
                            all_passed = False
                            message = "Your SQL is not correct."
                            break
                        except Exception:
                            all_passed = False
                            message = "Your SQL is not correct."
                            break
                    if all_passed:
                        passed = True
                        message = "SQL passed all test cases."

            pool.putconn(conn)

            if passed:
                if interact_mode == "c-interact":
                    reward = phase_rewards_first.get(current_phase, 0) if is_first_try else phase_rewards_debug.get(current_phase, 0)
                else:
                    reward = phase_rewards_first.get(current_phase, 0)
                if current_phase == 1:
                    _successful_phase1_sql[req_task_id] = req_sql
                    has_follow_up = bool(td.get("follow_up") and td["follow_up"].get("sol_sql"))
                    if has_follow_up:
                        # Rebuild from the canonical post-preprocess state, apply
                        # the accepted P1 SQL and official phase-transition cleanup,
                        # then freeze that state for all P2 attempts.
                        td["_snapshot_db"] = _create_phase1_snapshot(td, pred_sqls)
                        _submit_attempts[req_task_id][2] = 0
                        td["_current_phase"] = 2
                        follow_up_query = td["follow_up"].get("query", "Please complete the follow-up task.")
                        return SubmitSQLResponse(
                            passed=True, message=f"Phase 1 correct! (Reward: {reward}). Moving to Phase 2.",
                            reward=reward, phase_completed=1, has_follow_up=True,
                            follow_up_query=follow_up_query)
                    else:
                        return SubmitSQLResponse(
                            passed=True, message=f"Phase 1 correct! (Reward: {reward}). Task finished.",
                            reward=reward, phase_completed=1, has_follow_up=False)
                else:
                    return SubmitSQLResponse(
                        passed=True, message=f"Phase 2 correct! (Reward: {reward}). Task finished.",
                        reward=reward, phase_completed=2, has_follow_up=False)
            else:
                # Failed: restore task DB to pre-submit state for agent exploration
                if current_phase == 1:
                    reset_task_db(task_db, td["_initial_snapshot_db"])
                else:
                    snapshot = td.get("_snapshot_db")
                    if snapshot:
                        reset_task_db(task_db, snapshot)

                return SubmitSQLResponse(
                    passed=False, message=f"SQL failed Phase {current_phase}. {message}",
                    reward=0.0)
        except Exception as inner_e:
            try: pool.putconn(conn)
            except: pass
            raise inner_e
    except Exception as e:
        logger.error(f"Submit infrastructure error for {req_task_id}: {e}", exc_info=True)
        raise DatabaseStateError(
            f"Submit database state failure for {req_task_id}: {type(e).__name__}: {e}"
        ) from e


@app.post("/submit", response_model=SubmitSQLResponse)
async def submit_sql_endpoint(req: SubmitSQLRequest):
    td = _task_data.get(req.task_id)
    if not td: raise HTTPException(404, f"Task {req.task_id} not initialized")
    try:
        return await asyncio.to_thread(
            _submit_sql_sync, req.task_id, req.sql, td, _submit_attempts, _successful_phase1_sql
        )
    except DatabaseStateError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/schema")
async def get_schema(req: SchemaRequest):
    td = _task_data.get(req.task_id)
    if not td: raise HTTPException(404, f"Task {req.task_id} not initialized")
    db_name = td["selected_database"]
    _load_db_data(db_name)
    return {"schema": _schema_cache.get(db_name, "Schema not available")}


@app.post("/all_column_meanings")
async def get_all_column_meanings(req: SchemaRequest):
    td = _task_data.get(req.task_id)
    if not td: raise HTTPException(404, f"Task {req.task_id} not initialized")
    db_name = td["selected_database"]
    _load_db_data(db_name)
    return {"column_meanings": json.dumps(_column_meanings_cache.get(db_name, {}), indent=2)}


@app.post("/column_meaning")
async def get_column_meaning(req: ColumnMeaningRequest):
    td = _task_data.get(req.task_id)
    if not td: raise HTTPException(404, f"Task {req.task_id} not initialized")
    db_name = td["selected_database"]
    _load_db_data(db_name)
    key = f"{db_name}|{req.table_name.lower()}|{req.column_name.lower()}"
    meaning = _column_meanings_cache.get(db_name, {}).get(key, "Column meaning not found")
    return {"meaning": meaning if isinstance(meaning, str) else json.dumps(meaning)}


@app.post("/knowledge_names")
async def get_knowledge_names(req: SchemaRequest):
    td = _task_data.get(req.task_id)
    if not td: raise HTTPException(404, f"Task {req.task_id} not initialized")
    db_name = td["selected_database"]
    _load_db_data(db_name)
    agent_kb = _filter_knowledge(db_name, td)
    return {"names": list(agent_kb.keys())}


@app.post("/knowledge")
async def get_knowledge(req: KnowledgeRequest):
    td = _task_data.get(req.task_id)
    if not td: raise HTTPException(404, f"Task {req.task_id} not initialized")
    db_name = td["selected_database"]
    _load_db_data(db_name)
    agent_kb = _filter_knowledge(db_name, td)
    if req.knowledge_name:
        entry = agent_kb.get(req.knowledge_name)
        if entry:
            visible = {k: entry[k] for k in KNOWLEDGE_VISIBLE_FIELDS if k in entry}
            return {"knowledge": json.dumps(visible, indent=2)}
        return {"knowledge": "Knowledge not found."}
    else:
        visible_kbs = []
        for e in agent_kb.values():
            visible_kbs.append({k: e[k] for k in KNOWLEDGE_VISIBLE_FIELDS if k in e})
        return {"knowledge": json.dumps(visible_kbs, indent=2)}


def _cleanup_task_sync(database_names: List[str]):
    """Blocking cleanup — runs in thread pool."""
    _drop_databases(database_names, suppress_errors=False)


@app.post("/cleanup_task")
async def cleanup_task(req: SchemaRequest):
    td = _task_data.get(req.task_id)
    if not td:
        return {"status": "ok", "task_id": req.task_id}
    physical_names = td.get("_physical_db_names", {})
    database_names = [
        td.get("_snapshot_db"),
        physical_names.get("phase1"),
        physical_names.get("initial") if td.get("_initial_snapshot_owned") else None,
        td.get("_task_db"),
    ]
    try:
        await asyncio.to_thread(_cleanup_task_sync, database_names)
    except Exception as e:
        logger.warning(f"Cleanup failed for {req.task_id}: {e}")
    _task_data.pop(req.task_id, None)
    _submit_attempts.pop(req.task_id, None)
    _successful_phase1_sql.pop(req.task_id, None)
    return {"status": "ok", "task_id": req.task_id}


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "db_environment"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=settings.db_env_port)
