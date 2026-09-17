# 文档导航

[返回项目首页](../README.md)

## 当前版本

| 阅读目的 | 入口 |
| --- | --- |
| 了解方法与主要结果 | [项目 README](../README.md) |
| 理解四维信息、Check、Gate、Main 与 fallback | [框架与数据流](architecture.md) |
| 安装依赖、离线验证、准备真实评测 | [安装与运行](getting-started.md) |
| 查看 600 题成绩、成本与统计限制 | [2026 年 9 月版本说明](releases/2026-09-16/README.md) |
| 读取成绩数据 | [core_results.json](releases/2026-09-16/core_results.json) |
| 核对已评测源码及发布修复 | [candidate_manifest.json](releases/2026-09-16/candidate_manifest.json) |
| 了解测试与跳过项 | [测试说明](../tests/README.md) |
| 区分当前与历史脚本 | [脚本导航](../scripts/README.md) |
| 查看模型预设的作用 | [配置说明](../configs/README.md) |

目录 `releases/2026-09-16/` 沿用材料建立日期；代码于 2026-09-17 发布。目录中的成绩与源码指纹是固定记录，不随后续文档整理改变。

## 历史与上游

以下内容保留用于追溯，不作为当前版本的使用入口。

- [上游 BIRD-Interact-ADK 说明](upstream/README.md)：原架构、用法、Lite 结果和论文引用。原 `architecture.png / architecture.svg` 也是上游架构图。
- [初始源码快照](../baseline/README.md)：研究起点的来源、依赖与校验记录，不是当前 Full600 成绩目录。
- [阶段验收记录](../evidence/README.md)：早期研究过程，阶段通过不等于当前正式配置启用了该功能。
- [早期研究环境说明](../RESEARCH_ENVIRONMENT.md)：包含当时的本机路径、端口及隔离约定，不是通用部署配置。
- [非 ADK 基线提交说明](OFFICIAL_NONADK_GLM47_SUBMISSION.md)：另一条基线实验路线，不适用于当前 Valibra。
- [已退役实验断言](../tests/historical/README.md)：保留原内容，不计为当前通过的测试。

## 维护约定

运行代码保持现有包路径，避免为目录美观破坏导入或历史引用。新增方法文档放在 `docs/`，版本结果放在 `docs/releases/<日期>/`；原始轨迹、评测明细、凭据和数据库保留本地，不加入公共文档。
