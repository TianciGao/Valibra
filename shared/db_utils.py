"""Database utilities ported from BIRD-Interact evaluation code.
确针对 PostgreSQL 实现的 数据库工具函数集合，
主要用于执行 SQL 查询、管理数据库连接池、处理 SQL 查询结果等操作。

"""

import hashlib
import json
import logging
import os
import re
import subprocess
import threading
import time
from datetime import date, datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple

import psycopg2
import sqlparse
from psycopg2 import OperationalError
from psycopg2.pool import ThreadedConnectionPool

from shared.config import settings

logger = logging.getLogger(__name__)

PG_IDENTIFIER_MAX_BYTES = 63
_TASK_DB_NAMESPACE = "bird-interact-adk-task-db-v1"

_postgresql_pools: Dict[str, ThreadedConnectionPool] = {}
_pool_lock = threading.Lock()

_ANALYZE_TARGET_RE = re.compile(
    r'^\s*ANALYZE(?:\s*\([^)]*\))?\s+(?:ONLY\s+)?'
    r'(?:(?:"[^"]+"|[A-Za-z_][\w$]*)\.)?'
    r'(?P<table>"[^"]+"|[A-Za-z_][\w$]*)',
    re.IGNORECASE | re.DOTALL,
)


def _analyze_target_table(query: str) -> Optional[str]:
    """Return the relation named by a simple PostgreSQL ANALYZE statement."""
    match = _ANALYZE_TARGET_RE.match(query or "")
    if not match:
        return None
    return match.group("table").strip('"')


def _wait_for_analyze_visibility(query: str, conn, timeout: float = 2.0) -> None:
    """Wait until pg_stat_all_tables exposes a just-completed ANALYZE.

    PostgreSQL may publish cumulative statistics asynchronously.  The benchmark
    custom tests query last_analyze immediately, so without this small barrier a
    correct ANALYZE can pass or fail depending on timing and concurrency.
    """
    table_name = _analyze_target_table(query)
    if not table_name:
        return

    deadline = time.monotonic() + timeout
    while True:
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT pg_stat_clear_snapshot();")
            cursor.execute(
                "SELECT last_analyze IS NOT NULL "
                "FROM pg_stat_all_tables WHERE relname = %s;",
                (table_name,),
            )
            row = cursor.fetchone()
            conn.commit()
        finally:
            cursor.close()
        if row and row[0]:
            return
        if time.monotonic() >= deadline:
            return
        time.sleep(0.05)


def _get_or_init_pool(db_name: str) -> ThreadedConnectionPool: # 定义一个函数，用于获取或初始化指定数据库的连接池
    with _pool_lock:
        if db_name not in _postgresql_pools:
            _postgresql_pools[db_name] = ThreadedConnectionPool(
                settings.pg_minconn, settings.pg_maxconn,
                dbname=db_name, user=settings.pg_user,
                password=settings.pg_password, host=settings.pg_host,
                port=settings.pg_port,
            )
        return _postgresql_pools[db_name]


def close_pool(db_name: str): # 定义一个函数，用于关闭指定数据库的连接池
    with _pool_lock:
        if db_name in _postgresql_pools:
            pool = _postgresql_pools.pop(db_name)
            pool.closeall()


def perform_query(query: str, db_name: str, conn=None): # 定义一个函数，用于执行指定数据库的 SQL 查询，并返回查询结果、连接对象和游标描述
    MAX_ROWS = 10000
    if conn is None:
        pool = _get_or_init_pool(db_name)
        conn = pool.getconn()
    cursor = conn.cursor()
    cursor.execute("SET statement_timeout = '60s';")
    try:
        cursor.execute(query)
        # PostgreSQL's cursor metadata, rather than the first SQL keyword,
        # determines whether the statement produced a result set.  A writable
        # CTE starts with WITH but may intentionally have no RETURNING clause;
        # attempting fetchmany() in that case turns a successful write into a
        # spurious "no results to fetch" evaluator failure.
        if cursor.description is not None:
            rows = cursor.fetchmany(MAX_ROWS + 1)
            result = rows[:MAX_ROWS]
        else:
            result = None
        desc = cursor.description
        conn.commit()
        _wait_for_analyze_visibility(query, conn)
        return result, conn, desc
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cursor.close()


def execute_queries(queries, db_name: str, conn=None): # 定义一个函数，用于执行多个 SQL 查询，并返回结果、错误、超时和游标描述
    """Execute queries and return (result, error, timeout, cursor_description)."""
    if isinstance(queries, str):
        queries = [queries]
    if not queries:
        return None, None, False, None
    result = None
    desc = None
    for query in queries:
        if not query or not query.strip():
            continue
        try:
            result, conn, desc = perform_query(query, db_name, conn=conn)
        except psycopg2.errors.QueryCanceled:
            return None, None, True, None
        except (OperationalError, psycopg2.Error) as e:
            return None, str(e), False, None
        except Exception as e:
            return None, str(e), False, None
    return result, None, False, desc


def _pg_env() -> tuple: # 定义一个函数，用于获取 PostgreSQL 的连接参数和环境变量
    """Return (common_args, env_vars) for subprocess commands."""
    env_vars = os.environ.copy()
    env_vars["PGPASSWORD"] = settings.pg_password
    args = ["-h", settings.pg_host, "-p", str(settings.pg_port), "-U", settings.pg_user]
    return args, env_vars


def _validate_database_identifier(db_name: str, *, label: str) -> None:
    if not db_name:
        raise ValueError(f"{label} must not be empty")
    encoded_length = len(db_name.encode("utf-8"))
    if encoded_length > PG_IDENTIFIER_MAX_BYTES:
        raise ValueError(
            f"{label} exceeds PostgreSQL's {PG_IDENTIFIER_MAX_BYTES}-byte "
            f"identifier limit: {encoded_length} bytes"
        )


def task_database_names(base_db: str, task_id: str) -> Dict[str, str]:
    """Return deterministic, collision-resistant physical DB names for a task.

    Logical dataset/database identifiers remain unchanged everywhere outside the
    DB service.  Only transient PostgreSQL copies use these bounded names.
    Determinism lets the runner find and remove leftovers after a service restart.
    """
    identity = f"{_TASK_DB_NAMESPACE}\0{base_db}\0{task_id}".encode("utf-8")
    digest = hashlib.sha256(identity).hexdigest()[:32]
    names = {
        "task": f"bi_t_{digest}",
        "initial": f"bi_i_{digest}",
        "phase1": f"bi_p_{digest}",
    }
    for role, name in names.items():
        _validate_database_identifier(name, label=f"{role} task database name")
    if len(set(names.values())) != len(names):
        raise ValueError("Generated task database names are not distinct")
    return names


def _drop_and_create_db(db_name: str, template_db: str): # 定义一个函数，用于删除指定数据库（如果存在）并从模板数据库重新创建
    """Drop db_name (if exists) and recreate from template_db."""
    _validate_database_identifier(db_name, label="target database name")
    _validate_database_identifier(template_db, label="template database name")
    if db_name == template_db:
        raise ValueError(
            f"Refusing to recreate database {db_name!r} from itself; this would "
            "drop the template before cloning it"
        )
    args, env_vars = _pg_env()
    close_pool(db_name)
    subprocess.run(
        ["psql", *args, "-d", "postgres", "-c",
         f"SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = '{db_name}' AND pid <> pg_backend_pid();"],
        check=True, env=env_vars, timeout=60,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        ["dropdb", "--if-exists", *args, db_name],
        check=True, env=env_vars, timeout=60,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        ["createdb", *args, db_name, "--template", template_db],
        check=True, env=env_vars, timeout=60,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def reset_and_restore_database(db_name: str): # 定义一个函数，用于重置指定数据库并从模板数据库恢复
    template_db = f"{db_name}_template"
    _drop_and_create_db(db_name, template_db)


def clone_database(db_name: str, template_db: str) -> str:
    """Recreate an explicitly named transient database from a template."""
    _drop_and_create_db(db_name, template_db)
    return db_name


def create_task_db(base_db: str, task_id: str, template: str = None) -> str: # 定义一个函数，用于为每个任务创建一个独立的数据库副本，并返回任务数据库的名称
    """Create a per-task DB copy. Returns task DB name.

    Args:
        base_db: Logical base database name.
        task_id: Logical task identifier.
        template: Template DB to copy from. Defaults to {base_db}_template.
    """
    task_db = task_database_names(base_db, task_id)["task"]
    template_db = template or f"{base_db}_template"
    return clone_database(task_db, template_db)


def reset_task_db(task_db: str, template_source: str): # 定义一个函数，用于重置每个任务的数据库副本，从模板或快照数据库恢复
    """Reset a per-task DB from a template/snapshot DB."""
    _drop_and_create_db(task_db, template_source)


def drop_task_db(task_db: str): # 定义一个函数，用于删除每个任务的数据库副本，并关闭其连接池
    """Drop a per-task DB and close its connection pool."""
    _validate_database_identifier(task_db, label="task database name")
    args, env_vars = _pg_env()
    close_pool(task_db)
    subprocess.run(
        ["psql", *args, "-d", "postgres", "-c",
         f"SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = '{task_db}' AND pid <> pg_backend_pid();"],
        check=True, env=env_vars, timeout=60,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    subprocess.run( # 使用 subprocess.run() 方法调用 dropdb 命令删除指定的任务数据库，并设置超时时间为 60 秒，同时将标准输出和标准错误输出重定向到 DEVNULL，以避免在控制台显示
        ["dropdb", "--if-exists", *args, task_db],
        check=True, env=env_vars, timeout=60,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def get_connection_for_phase(db_name: str): # 定义一个函数，用于获取指定数据库的连接对象
    pool = _get_or_init_pool(db_name)
    return pool.getconn()


def process_decimals_recursive(item, decimal_places: int): # 定义一个函数，用于递归处理数据结构中的 Decimal 类型，将其四舍五入到指定的小数位数
    quantizer = Decimal(1).scaleb(-decimal_places)
    if isinstance(item, Decimal):
        return item.quantize(quantizer, rounding=ROUND_HALF_UP)
    elif isinstance(item, float):
        return round(item, decimal_places)
    elif isinstance(item, (list, tuple)):
        return type(item)(process_decimals_recursive(x, decimal_places) for x in item)
    elif isinstance(item, dict):
        return {k: process_decimals_recursive(v, decimal_places) for k, v in item.items()}
    return item


def preprocess_results(results, decimal_places: int = 2): # 定义一个函数，用于预处理 SQL 查询结果，将其中的日期、时间和 Decimal 类型的数据进行格式化和四舍五入处理，并返回处理后的结果列表
    if results is None:
        return []
    processed = []
    for row in results:
        processed_row = []
        for item in row:
            if isinstance(item, (date, datetime)):
                processed_row.append(item.strftime("%Y-%m-%d"))
            else:
                pi = process_decimals_recursive(item, decimal_places)
                if isinstance(pi, (dict, list)):
                    processed_row.append(json.dumps(pi, sort_keys=True))
                else:
                    processed_row.append(pi)
        processed.append(tuple(processed_row))
    return processed


def remove_comments(sql_list: List[str]) -> List[str]: # 定义一个函数，用于移除 SQL 查询中的注释，包括块注释和行注释，并返回处理后的 SQL 查询列表
    cleaned = []
    for sql in sql_list:
        no_block = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
        no_line = re.sub(r"--.*?(\r\n|\r|\n)", r"\1", no_block)
        no_blank = re.sub(r"\n\s*\n+", "\n", no_line)
        cleaned.append(no_blank.strip())
    return cleaned


def remove_distinct(sql_list: List[str]) -> List[str]: # 定义一个函数，用于移除 SQL 查询中的 DISTINCT 关键字，并返回处理后的 SQL 查询列表
    """Remove ordinary DISTINCT while preserving PostgreSQL DISTINCT ON.

    The benchmark intentionally ignores duplicate-only differences for normal
    SELECT DISTINCT queries.  DISTINCT ON, however, is a PostgreSQL selection
    construct: deleting only DISTINCT changes valid ``SELECT DISTINCT ON`` into
    invalid ``SELECT ON`` and makes even the ground truth fail against itself.
    """
    ordinary_distinct = re.compile(r"\bDISTINCT\b(?!\s+ON\s*\()", re.IGNORECASE)
    return [ordinary_distinct.sub("", query) for query in sql_list]


def split_sql_statements(sql: str) -> List[str]:
    """Split a submitted SQL payload without breaking quoted function bodies.

    The HTTP tool accepts one string, while Full management test cases receive a
    list of statements and may address individual entries.  ``sqlparse.split``
    preserves semicolons inside quoted strings and PostgreSQL dollar-quoted
    function bodies, unlike a raw ``str.split(';')``.
    """
    if not isinstance(sql, str) or not sql.strip():
        return []
    return [statement.strip() for statement in sqlparse.split(sql) if statement.strip()]


def _remove_round_functions(sql_string: str) -> str: # 定义一个函数，用于移除 SQL 查询中的 ROUND 函数，并返回处理后的 SQL 查询字符串
    def find_matching_paren(text, start):
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "(": depth += 1
            elif text[i] == ")":
                depth -= 1
                if depth == 0: return i
        return -1

    def find_first_arg_end(text, start): # 定义一个函数，用于查找 SQL 查询中 ROUND 函数的第一个参数的结束位置，并返回该位置的索引
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "(": depth += 1
            elif text[i] == ")":
                if depth == 0: return i
                depth -= 1
            elif text[i] == "," and depth == 0: return i
        return len(text)

    result = sql_string # 定义一个变量，用于存储处理后的 SQL 查询字符串
    while True:
        match = re.search(r"ROUND\s*\(", result, re.IGNORECASE)
        if not match: break
        start = match.start()
        open_p = match.end() - 1
        first_end = find_first_arg_end(result, open_p + 1)
        close_p = find_matching_paren(result, open_p)
        if close_p == -1: break
        first_arg = result[open_p + 1: first_end].strip()
        result = result[:start] + first_arg + result[close_p + 1:]
    return result


def remove_round(sql_list: List[str]) -> List[str]: # 定义一个函数，用于移除 SQL 查询中的 ROUND 函数，并返回处理后的 SQL 查询列表
    return [_remove_round_functions(sql) for sql in sql_list]


def ex_base(pred_sqls, sol_sqls, db_name, conn, conditions=None) -> int: # 定义一个函数，用于比较预测 SQL 查询和标准答案 SQL 查询的结果是否一致，并返回 1 表示一致，0 表示不一致
    if not pred_sqls or not sol_sqls:
        return 0
    pred_res, pred_err, pred_to, _ = execute_queries(pred_sqls, db_name, conn)
    gt_res, gt_err, gt_to, _ = execute_queries(sol_sqls, db_name, conn)
    if any([pred_err, pred_to, gt_err, gt_to]):
        return 0
    pred_res = preprocess_results(pred_res)
    gt_res = preprocess_results(gt_res)
    if not pred_res or not gt_res:
        return 0
    if conditions and conditions.get("order", False):
        return 1 if pred_res == gt_res else 0
    return 1 if set(pred_res) == set(gt_res) else 0


def test_case_default(pred_sqls, sol_sqls, db_name, conn, conditions=None): # 定义一个函数，用于测试预测 SQL 查询和标准答案 SQL 查询的结果是否一致，并返回 1 表示一致，0 表示不一致
    pred_sqls = remove_round(remove_distinct(remove_comments(pred_sqls)))
    sol_sqls = remove_round(remove_distinct(remove_comments(sol_sqls)))
    result = ex_base(pred_sqls, sol_sqls, db_name, conn, conditions)
    assert result == 1, f"ex_base returned {result} but expected 1."
    return result
