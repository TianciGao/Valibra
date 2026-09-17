# 测试说明

[项目首页](../README.md) · [发布检查记录](../docs/releases/2026-09-16/release_preflight.json)

## 运行

在安装项目依赖及 pytest 后，从仓库根目录执行：

```bash
python -m pytest -q -p no:cacheprovider tests
```

当前套件是离线验证，不调用真实 Provider、数据库或官方工具。测试中使用的 SQL、工具返回与凭据字符串是测试输入，不是发布的完整评测轨迹。

## 覆盖范围

| 范围 | 主要位置 |
| --- | --- |
| 四维状态、表达式与字段授权 | `valibra/test_sg1_*`、`test_sg2_*` |
| Provider 解析、阶段输入与失败边界 | `valibra/test_sg4_*`、`test_sql_grounding_*` |
| Mapping 未解决概念与受限修正 | `valibra/test_mapping_*` |
| Check、澄清、Gate 与草稿隔离 | `valibra/test_check_*`、`test_final_*`、`test_clarification_*` |
| P2 知识保留与证据复用 | `valibra/test_p2_*` |
| Main 交接与官方 fallback | `valibra/test_main_*`、`test_ainteract_fallback.py` |
| 数据库环境辅助逻辑 | `test_db_environment_state.py` |
| 已评测源码与发布修复边界 | `valibra/test_release_candidate_contract.py` |

文件名保留阶段演进痕迹。测试某个默认关闭的模块，不意味着正式配置已启用该实验。

## 跳过项与历史文件

发布时，带本地数据的环境为 **513 通过、5 跳过**；不含数据集的发布副本为 **511 通过、7 跳过**。两者均有 315 个子测试通过。

- 四项旧 Clarification 测试沿用原 skip。
- 一项历史 SG7 评测测试缺少退役 P7 依赖时显式 skip。
- 两项源数据完整性检查，仅在 Full600 输入或对应知识库不存在时 skip；本地数据齐全时正常执行。
- [historical/](historical/README.md) 保存两份退役 Prompt 的措辞断言，故意不使用 `test_*.py` 名称，不纳入当前套件。

这些记录不是新 E2E，也不能替代模型语义能力或最终 benchmark 成绩。
