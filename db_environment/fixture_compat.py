"""Audited compatibility fixes for inconsistent Full benchmark fixtures.

These fixes preserve the task intent and the official GT SQL.  They only repair
environment/test code that cannot execute correctly as published.  Keeping the
exceptions in one small module makes the research fork explicit and hashable.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional


# The published cleanup for crypto_exchange_M_1 references non-existent,
# incorrectly cased columns.  P1 deletes rows from orderExecutions after a full
# backup is created by preprocess_sql.  Restoring missing primary-key rows from
# that backup is the intended inverse and produces the canonical P2 start state.
_ENVIRONMENT_SQL_OVERRIDES = {
    ("crypto_exchange_M_1", "clean_up_sqls"): [
        '''
        INSERT INTO "orderExecutions"
        SELECT backup.*
        FROM "orderexecutions_bak" AS backup
        ON CONFLICT ("RecordVault") DO NOTHING;
        '''
    ],
}


def environment_sql(
    task_id: str,
    label: str,
    original: Any,
) -> Any:
    """Return an audited environment-SQL correction, if one is registered."""
    return _ENVIRONMENT_SQL_OVERRIDES.get((task_id, label), original)


def run_custom_test_override(
    *,
    task_id: str,
    phase: int,
    test_index: int,
    pred_sqls: List[str],
    db_name: str,
    conn: Any,
    execute_queries: Callable[..., Any],
) -> bool:
    """Run an audited replacement for a time-dependent custom test.

    Returns True when an override handled the test, otherwise False.
    """
    if (task_id, phase, test_index) != ("solar_panel_M_5", 1, 0):
        return False

    execute_queries("DROP FUNCTION IF EXISTS get_plant_age(text);", db_name, conn)
    execute_queries(pred_sqls, db_name, conn)
    result, error, timed_out = execute_queries(
        '''
        SELECT
            get_plant_age('SP9227')::real,
            EXTRACT(YEAR FROM AGE(CURRENT_DATE, goliveon))::real
        FROM plants
        WHERE sitekey = 'SP9227';
        ''',
        db_name,
        conn,
    )
    try:
        assert not error and not timed_out, error or "age validation timed out"
        assert result and len(result) == 1, "plant SP9227 was not found"
        actual, expected = result[0]
        assert actual == expected, (
            f"Function returned age {actual}, expected {expected} for CURRENT_DATE"
        )
    finally:
        execute_queries("DROP FUNCTION IF EXISTS get_plant_age(text);", db_name, conn)
    return True
