# 安装与运行

[项目首页](../README.md) · [文档导航](README.md) · [框架说明](architecture.md)

先做离线验证，再配置真实评测。下面的启动示例是当前代码的接入方式，不承诺重现同一组随机模型输出，也不替代已评测版本的[配置清单](releases/2026-09-16/candidate_manifest.json)。

## 1. 取得代码与安装依赖

推荐使用 Linux / WSL 和独立 Python 环境；发布检查使用 Python 3.12。

```bash
git clone --branch research/sql-grounding-v1 https://github.com/TianciGao/Valibra.git
cd Valibra
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install pytest pytest-subtests
```

已有工作目录时不要重复创建或覆盖环境。运行依赖见 [requirements.txt](../requirements.txt)；`baseline/dependencies.freeze.txt` 是早期快照，不是当前发布环境锁文件。

## 2. 离线验证

```bash
python -m pytest -q -p no:cacheprovider tests
```

这套测试不发起 Provider、数据库或官方工具调用。缺少数据集时，两项源数据完整性检查会明确跳过；四项旧澄清测试及一项缺少历史依赖的 SG7 测试也不计为通过。详见[测试说明](../tests/README.md)。

只阅读代码、查看结果或运行这些测试，不需要启动 Docker 或填写 API key。

## 3. 准备真实评测

真实评测会调用模型、消耗费用，并创建和修改任务数据库。运行前需要：

- 与评测版本相符的任务、schema、字段说明、知识库及所需评测文件；
- BIRD-Interact Full PostgreSQL 数据库；
- Main/Grounding 模型和用户模拟器的有效访问凭据；
- 独立端口、足够磁盘空间，以及新建的任务会话。

数据目录由 `DATASET` 决定；Full 默认读取 `bird-interact-full/bird_interact_data.jsonl` 及对应数据库元数据。数据集及参考评测文件不随仓库分发，其获取方式见[上游说明](upstream/README.md#dataset)。

先检查端口和容器，再启动本项目隔离数据库：

```bash
bash scripts/start_research_db.sh
```

该脚本对应 [docker-compose.research.yml](../docker-compose.research.yml)，将数据库绑定到本机 `6433`。这是实际启动命令，不属于离线验证；不要在不清楚现有容器和数据卷用途时执行清理或删除操作。

## 4. 区分三类配置

| 配置 | 入口 | 注意 |
| --- | --- | --- |
| Main 模型与生成参数 | `MODEL_PRESET`、`SYSTEM_AGENT_*` | 预设与显式生成参数冲突会报错 |
| Grounding 模型与连接 | `GROUNDING_*` | 独立配置，不会自动借用 Main 凭据 |
| 用户模拟器 | `USER_SIM_*` | 模型、端点、协议策略独立设置 |

[.env.example](../.env.example) 只用于查看已有变量，不是 Full600 的完整复现配置。已有 `.env` 不应被覆盖，也不要把凭据提交到 Git。

以下为已发布代码对应的模型和 Grounding 参数示例。在干净 shell 中配置；不要同时导出与预设冲突的 `SYSTEM_AGENT_MAX_TOKENS` 等生成参数：

```bash
export MODEL_PRESET=glm52_high_32768
export GROUNDING_MODEL_PRESET=glm52_high_32768
export GROUNDING_UPDATER_MODE=llm
export GROUNDING_TIMEOUT_SECONDS=600
export GROUNDING_MAX_TOKENS=32768
export GROUNDING_MAX_CALLS_PER_TASK=32
export GROUNDING_PROMPT_SHA256=abcd64292037ba6fa5f6672c04383d47f9742da0ae63763afd66cc4ee8affccd
```

主模型与 Grounding 的 API 地址、凭据文件需分别配置：

```bash
export SYSTEM_AGENT_API_BASE=https://open.bigmodel.cn/api/paas/v4
export GROUNDING_API_BASE=https://open.bigmodel.cn/api/paas/v4
export SYSTEM_AGENT_API_KEY_FILE=/absolute/path/to/main-key
export GROUNDING_API_KEY_FILE=/absolute/path/to/grounding-key
```

以上路径是占位符，不是仓库文件。不要同时为 Grounding 设置 `GROUNDING_API_KEY` 和 `GROUNDING_API_KEY_FILE`。用户模拟器还需填写自己的 `USER_SIM_MODEL / API_BASE / API_KEY_FILE`，并按端点选择认证方式；本次评测的模拟器模型为 `anthropic/claude-haiku-4-5-20251001`。

运行模式和启用功能必须显式对齐：

```bash
export DATASET=full
export PROMPT_VERSION=v2
export PATIENCE=3
export USER_SIM_PROTOCOL_POLICY=official
export USER_SIM_PROTOCOL_MAX_ATTEMPTS=1
export USER_SIM_DISABLE_THINKING=true
export VALIBRA_EXECUTION_PROFILE=leaderboard

export VALIBRA_AINTERACT_FALLBACK=1
export VALIBRA_ANSWER_CONTRACT_GUARD_V0=1
export VALIBRA_ANSWER_CONTRACT_SHADOW=1
export VALIBRA_FINAL_REGROUNDING_GATE=1
export VALIBRA_MAIN_FRESH_RECOMPILE_R1=1
export VALIBRA_P2_CHECK_CUMULATIVE_OFFICIAL_EVIDENCE=1
export VALIBRA_P2_EXACT_OFFICIAL_EVIDENCE_REUSE=1

export VALIBRA_CHECK_RESOLVED_LITERAL_EXECUTABLE_CARRIER_R1=0
export VALIBRA_CHECK_RESOLVED_LITERAL_PROPOSAL_SHADOW_R1=0
```

`research` 与 `leaderboard` 是两种执行配置，不是 stress/free 的同义词。默认值是 `research`；本次 600 题用的是 `leaderboard`。以服务 `/health` 返回的有效配置为准。

## 5. 启动服务

当前接入需要 Valibra、用户模拟器和数据库环境三个 HTTP 服务。建议各开一个终端；每个终端都应使用同一虚拟环境和上述配置。调度器终端也须设置相同端口。

| 服务 | Python 入口 | 示例端口 |
| --- | --- | ---: |
| Valibra | `valibra_agent.server:app` | 6110 |
| 用户模拟器 | `user_simulator.server:app` | 6101 |
| 数据库环境 | `db_environment.server:app` | 6102 |
| PostgreSQL | 隔离 Docker 实例 | 6433 |

设置连接参数。数据库用户名和密码应与实际部署一致：

```bash
export SYSTEM_AGENT_PORT=6110
export USER_SIM_PORT=6101
export DB_ENV_PORT=6102
export PG_HOST=127.0.0.1
export PG_PORT=6433
# 另行设置 PG_USER 和 PG_PASSWORD，与数据库配置一致。
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
```

分别执行：

```bash
python -m uvicorn db_environment.server:app --host 127.0.0.1 --port 6102
```

```bash
python -m uvicorn user_simulator.server:app --host 127.0.0.1 --port 6101
```

```bash
python -m uvicorn valibra_agent.server:app --host 127.0.0.1 --port 6110
```

这里不用 `scripts/start_valibra_services.sh` 启动付费评测：它会加载早期 `research_env.sh`，重置部分凭据并将 Main/模拟器指向离线端点，也不会自动关闭另行设置的 Grounding 连接。该脚本不是当前真实评测的一键入口。

服务启动后检查：

```bash
curl --noproxy '*' --fail http://127.0.0.1:6110/health
curl --noproxy '*' --fail http://127.0.0.1:6101/health
curl --noproxy '*' --fail http://127.0.0.1:6102/health
```

Valibra 返回的 `service` 应为 `valibra_agent`。还需确认 `adk_available`、`execution_profile`、`grounding_provider_enabled`、模型预设及端口；HTTP 返回 healthy 不等于模型访问和数据库内容都已验证。

## 6. 从一个任务开始

以下命令会正式运行任务并产生费用。先完成配置检查，再执行：

```bash
python -m orchestrator.runner \
  --mode a-interact \
  --data bird-interact-full/bird_interact_data.jsonl \
  --limit 1 \
  --concurrency 1 \
  --output results/valibra_smoke.json
```

确认任务运行正常后，再按预先冻结的清单扩大评测。不要依据单题结果临时换题或只汇总成功的重跑；必须记录版本、任务顺序、替代运行及 token 口径。

`c-interact` 在 Valibra 构建入口会回到原官方 Agent，不属于本次四维框架成绩；`oracle` 使用参考 SQL 检查评测链路，也不能计为模型成绩。
