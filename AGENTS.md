# Research Repository Rules

- SQL Grounding V1 的唯一开发分支固定为 `research/sql-grounding-v1`；SG1–SG7 只向 `origin/research/sql-grounding-v1` 推送。
- annotated tag `p7.1d-pass` 是旧 Requirement Grounding 的不可变 rollback baseline，固定指向 `22dbf71de443843ec2945fc929022e79f50a3d2e`，不得移动、删除、重建或重写。
- `research/main` 是集成分支，已经通过 PR #1 合入 SG0；它不是旧框架的冻结 rollback 分支。SQL Grounding 阶段不得直接 push、merge 或创建面向 `research/main` 的 PR。
- 每次完成一个明确授权的任务后，自动执行检查、提交并推送，不需要再次询问是否提交。
- “完成”是指该任务相关测试通过、差异已经检查，并且没有已知失败。
- 每个 SG 阶段必须单独验收；未经用户明确授权，不得进入下一阶段。
- 只能暂存本次任务相关文件；禁止未经检查执行 `git add -A`。
- 工作树存在无关修改时不得自动提交，必须先报告。
- 禁止提交 `.env`、真实 API Key、Token、私钥、密码或 Provider 原始凭据。
- 禁止提交数据库、dump、数据卷、虚拟环境、缓存、完整评测结果、模型输出或私有大日志。
- 每次提交前必须检查 diff、测试结果、敏感信息和大文件。
- 提交信息使用简洁的 Conventional Commit 格式。
- 测试失败、任务未完成或存在安全疑点时，不提交、不推送。
- 完成任务后自动推送到 `origin/research/sql-grounding-v1`。
- 禁止 force push，禁止重写已经推送的历史。
- 普通任务只执行 commit 和 push；只有阶段明确通过时才创建 annotated tag。
- SQL Grounding V1 阶段标签固定为 `sqlg-v1-sg0-pass`、`sqlg-v1-sg1-pass`、`sqlg-v1-sg2-pass` 等。
- 推送失败时保留本地提交并报告，不进行破坏性恢复。

## SQL Grounding V1 Migration Boundary

- 旧 Runtime key 固定为 `valibra:grounding_runtime`；新 Runtime key 固定为 `valibra:sql_grounding_runtime`。
- 新旧 Runtime 不做自动迁移；旧结果只读归档，新实验必须创建新 Session。
- `valibra_agent/requirement_grounding/` 暂时只读保留：不得删除，也不得继续扩展旧 Requirement Grounding 语义。
- 每个 SG 阶段只能实现当轮明确授权的范围；不得提前创建或实现后续阶段的 SQL Grounding 业务代码。
