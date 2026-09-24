# 脚本导航

[项目首页](../README.md) · [当前安装与运行](../docs/getting-started.md)

这个目录同时保留了环境管理与历史评测脚本。文件名里的 `full`、`frozen` 或题号范围，并不表示它就是当前发布版入口。

## 当前代码的推荐入口

| 工作 | 入口 |
| --- | --- |
| 离线测试 | `python -m pytest -q -p no:cacheprovider tests` |
| 历史 Main 交接审计 | [audit_main_handoff.py](audit_main_handoff.py)，只读本地归档，不调用模型或数据库 |
| 启动隔离 PostgreSQL | [start_research_db.sh](start_research_db.sh)，先核对容器与端口 |
| 启动当前 HTTP 服务 | 按[运行指南](../docs/getting-started.md)显式启动三个 Python 模块 |
| 调度评测 | `python -m orchestrator.runner`，连接已核验的 Valibra 端口 |
| 检查模型预设 | [dry_run_model_preset.py](dry_run_model_preset.py)，先查看帮助与输入参数 |

## 离线交接审计

提供本地最终计分尝试索引（每行包含 `task`、`candidate_source`、`baseline`、`candidate`、`primary_p1`、`primary_full`、`fallback`、`replaced`），从仓库根目录运行：

```bash
python scripts/audit_main_handoff.py --root . --index /path/to/task_comparison.json --summary-only
```

脚本读取索引指向的 `official_result.json` 和 `provider_audits/`，按每题每阶段的首次真实 SQL Writer 请求检查摘要省略标记，并比对在此之前同题已取得的、带 enum 标记的字段说明。只打印统计、任务标识和哈希；不输出原始 Prompt、SQL 或参考答案。不加 `--summary-only` 可输出逐题诊断计数。

枚举字符串未出现只是字面覆盖诊断，不等于语义遗漏或失败原因；不同结果组不是因果对照。该工具不恢复旧 Session、不更改正式流程，也不构成新的模型或端到端评测。原始归档和逐题诊断应保留本地，不提交到仓库。

## 有界单轮交接实验

`probe_main_handoff.py` 是独立诊断工具，不是正式 Agent 或端到端评测。它固定比较 4 个归档字面缺口案例与 2 个历史成功控制：A 为首次 P1 Main 原始输入，B 仅追加在此之前同题已取得的全部可解析映射字段说明。每组一次、合计最多 12 次模型请求，无自动重试、不执行返回的工具调用或 SQL、不恢复历史 Runtime。

先离线准备，再在取得付费授权后显式执行：

```bash
python scripts/probe_main_handoff.py --output research-runtime/NEW_EXPERIMENT --prepare-index /path/to/task_comparison.json
python scripts/probe_main_handoff.py --output research-runtime/NEW_EXPERIMENT --run-paid
```

沿用归档 GLM-5.2 参数（每次输出上限 32,768 token），使用当前匹配模型的 HTTPS 连接配置。原始输入、证据来源哈希与响应仅写入忽略目录；启动后不支持重跑同一目录，避免重复计费。单轮候选的变化不等于数据库通过、端到端收益或统计显著改善。

## 环境管理脚本

- `research_env.sh`：早期研究环境隔离设置。会清理或覆盖部分 Provider 变量并将 Main/模拟器指向离线端点；不应在配置好真实评测后再次加载，也不能代替 Grounding 配置审查。
- `start_valibra_services.sh / stop_valibra_services.sh`：早期 Valibra 服务管理，固定端口和本地虚拟环境；启动会加载上述环境脚本。
- `start_research_services.sh / stop_research_services.sh`：早期官方服务的隔离运行，不等同于当前 Valibra。
- `start_research_db.sh / stop_research_db.sh`：隔离数据库管理。停止服务前先确认没有评测正在使用。
- `research_preflight.sh / research_production_guard.sh`：早期隔离与保护检查，不是当前 Full600 发布验收的替代品。

## 历史评测与数据辅助

`run_frozen_*`、`resume_frozen_*`、带固定模型名或题号的 `run_*`、`prepare_tasks_0501_0600.py` 属于特定历史批次。一些脚本包含固定路径、端口或 Provider 设置，不应直接拿来开始新评测。

`run_official_nonadk_*`、`preflight_official_nonadk.py`、`validate_nonadk_submission.py` 对应非 ADK 基线，不是 Valibra。

`combine_public_with_gt.py`、`prepare_task_range.py`、`reevaluate_saved_predictions_no_llm.py` 等处理数据或已有结果。名称中的 no_llm 只说明不调用模型，不保证不访问或改变数据库；使用前仍需核对脚本及输出位置。

上游 `start_services.sh / run_eval.sh` 含旧端口及进程管理逻辑，并受研究目录保护检查限制。本次整理保留原脚本，没有解除保护、迁移路径或执行任何启动/停止命令。
