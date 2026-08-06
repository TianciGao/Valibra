# P0 Service Lifecycle Review — Persistent PTY

**Result: P0 PASS**

All start, readiness, service verification, stop, and cleanup commands ran in the same persistent PTY session (session 1715).

Research PostgreSQL reached both required ready conditions within the 300-second bound. The three HTTP services then survived the start script return. At both the 10-second check and the additional 30-second check:

- all three PIDs existed and were non-defunct;
- command lines and Research working directories were correct;
- 6100, 6101, and 6102 were listening under the expected PIDs;
- all three health endpoints returned HTTP 200;
- PostgreSQL remained healthy and host-ready.

The existing exact stop scripts were used. After a 5-second graceful shutdown allowance, all three PIDs and listeners were gone. PostgreSQL, its container, and its network were stopped; the named Research volume was preserved.

One initial cleanup diagnostic matched its own Python scanner because the command text contained the word "uvicorn". The corrected self-excluding scan and the independent post-PTY scan found no Research uvicorn. This was a diagnostic false positive, not a lifecycle failure.

Formal ports, the formal PostgreSQL container identity/start time, and the formal results fingerprint/count are unchanged. Service logs contain only health calls and graceful shutdown; no `/run_session`, Provider, model, or evaluation activity occurred.

No business code, script, configuration, dependency, or Git commit was changed. Only this evidence directory was created.

