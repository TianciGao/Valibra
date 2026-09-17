# 发布前测试核对

## 已同步内容

第一阶段只修改测试和发布材料，17 个已评测运行文件的 SHA-256 均未变化。
修改前的测试已完整备份到本地 `research-runtime/`，不会上传该私有目录。

| 旧测试问题 | 处理 | 保留的检查 |
| --- | --- | --- |
| 固定早期 Prompt/Form/config 哈希 | 更新为 Full600 已评测版本的固定值 | 仍严格比较哈希；新增 17 个运行源码逐文件校验 |
| Mapping、Knowledge、Check 输入缺少 `unresolved_mappings` | 明确加入空列表 | 缺少该字段的输入仍必须拒绝，不修改 runtime 白名单 |
| Mapping 模拟响应仍返回旧的完整 State | 返回当前要求的 `MappingGroundingResponse` | Mapping ownership 和状态变更检查不变 |
| 测试期待旧的 metadata 内部键 | 改为当前 canonical `table.column` 键 | 元数据和字段授权检查不变 |
| 两份测试仅断言已退役的实验措辞 | 原文移至 `tests/historical/` | 当前 ownership、澄清和完整性测试保留；检查旧措辞没有重新启用 |
| SG7 历史评测脚本依赖缺失 | 依赖不存在时明确报告 skip | 不伪造通过，不改 SG7 脚本或补做历史评测 |

现有四个旧 Clarification 测试本来就标记为 skip，本次未调整其状态。

## 第一阶段发现的问题

两个 P2 流程测试当时仍然失败，并非继续同步断言就能解决：

| 测试 | P2 三阶段合并审计大小 | 当前上限 |
| --- | ---: | ---: |
| `test_p1_success_follow_up_grounds_once_without_rebootstrap` | 4,100 字节 | 4,096 字节 |
| `test_p1_failure_then_success_enters_p2_incremental` | 4,112 字节 | 4,096 字节 |

P2 状态推进本身成功，但合并后的记录被 `_require_bounded_audit` 拒绝，
`after_tool_exact_audit` 只留下一个 `ValueError` 错误摘要，原审计记录未保存。
测试随后无法读取应有的 P2 审计信息。

以下行号对应修复前的冻结源码：

- `valibra_agent/grounding_callbacks.py:199`：4,096 字节上限。
- `valibra_agent/grounding_callbacks.py:3009`：汇总 P2 各阶段审计。
- `valibra_agent/grounding_callbacks.py:2586`：嵌入 submit 的审计记录。
- `valibra_agent/grounding_callbacks.py:2643`：按工具调用 ID 保存审计，异常时记录错误摘要。
- `valibra_agent/grounding_callbacks.py:10268`：检查序列化后的字节数。

只读抽查了三个较小的历史 Primary P1 成功归档，均有同一
`after_tool_exact_audit / ValueError` 签名，且三个任务的 P1/P2 最终均通过。
历史错误摘要不含具体 exception message，因此不能仅凭签名证明每次都是同一原因，
也不能由这三个样本推算整个 600 题的发生率。

当前证据支持“审计记录可能丢失”，不支持把它说成“P2 执行失败”，
也不支持因此否定已有成绩。

## 2026-09-17 授权修复

用户单独批准了审计存储修复。没有修改原两项失败断言、缩短测试数据或增加 skip/xfail；修复不触及 Prompt、工具调用、预算、路由或四维状态。

最终采用原格式分段限额，而非增加引用层：普通记录和 control 摘要仍限 4,096 字节；P2 外层摘要和最多四个有序阶段各限 4,096 字节。整条工具记录限 24,832 字节，记录条数上限仍为 64。写入、读取和研究轨迹附加路径使用同一检查，不截断、不改写审计内容。

`grounding_callbacks.py` 与评测版的 AST 差异仅限两个限额常量和四个审计存储/验证函数。完整差异见 [audit_storage.patch](audit_storage.patch)。原评测哈希与发布哈希分别记录在清单中；其余 16 个运行文件和四项 Prompt/Form/配置指纹均保持一致。

验证结果：

- 原失败所在模块与新增审计测试：22 通过，14 个子测试通过。
- 全套离线测试：513 通过、0 失败、5 跳过，315 个子测试通过。
- 新增边界测试：逐阶段超限、外层超限、阶段数量/顺序、嵌套、UTF-8 字节数、存储篡改、64 条上限及失败时不部分写入。

已验证补丁可以反向应用；私有回滚包保留在本地。发布没有重新运行 600 题，已有成绩不冒充审计修复后的新评测。

## 不含数据集的发布副本

仅用拟提交文件导出的副本，不含本地 Full600 输入和知识库。两个源数据完整性测试原先会因文件缺失报错；现在只在所需文件不存在时明确 skip，原断言不变。本机有数据时继续执行，两项均通过。其余模拟数据测试不依赖私有文件。

未修改冻结源码中的三个历史文件末尾空行；补丁文件也保留 unified diff 必需的空白上下文。它们是已审查的格式告警，不为消除告警而改变已评测文件哈希。
