# P0 Service Lifecycle Bounded Retry

**Result: P0 FAIL**

The retry corrected the previous database readiness limitation: the existing named volume was preserved, the Research PostgreSQL container reached `healthy`, and host `pg_isready` returned `accepting connections` in the same observation within the 300-second bound.

The existing service start script then launched B0, User Simulator, and DB Environment. Its own health checks reached HTTP 200, as recorded in all three service logs. However, when the transient command shell returned, all three background uvicorn PIDs became defunct. The required explicit post-script health calls to 6100, 6101, and 6102 therefore returned connection refused. Because these are hard checks, P0 cannot pass.

No restart or alternative launch mode was attempted. Cleanup used the existing exact PID and database stop scripts. Research ports are free; no Research PID, uvicorn, container, or network remains; the named database volume is preserved. Runtime logs were moved intact into this evidence directory.

Formal state is unchanged: HTTP ports 6000/6001/6002 remain closed, PostgreSQL 5433 remains accepting, the formal PostgreSQL container identity/start time is unchanged, and the formal results fingerprint and file count remain unchanged.

No business code, script, configuration, dependency, model, Provider, evaluation, or Git commit action occurred.

