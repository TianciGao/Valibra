# 配置说明

[项目首页](../README.md) · [安装与运行](../docs/getting-started.md)

## 当前模型预设

`model_presets/` 保存模型生成参数，不保存 API key 或服务启动配置。

| 预设 | 用途 |
| --- | --- |
| [glm52_high_32768.json](model_presets/glm52_high_32768.json) | 本次 Full600 使用的 GLM-5.2 预设 |
| [glm52_max_65536.json](model_presets/glm52_max_65536.json) | 另一组 GLM-5.2 参数，不与本次成绩混用 |
| [glm47_matched_32768.json](model_presets/glm47_matched_32768.json) | 历史 GLM-4.7 对照配置 |

预设中的 `history` 是来源记录，不会自动将当前任务切换为其中的历史批次。

Main 通过 `MODEL_PRESET` 选择预设；Grounding 通过 `GROUNDING_MODEL_PRESET` 独立选择。连接参数、凭据、服务端口、执行 profile 与功能开关需另行设置。

## 发布配置

[当前发布清单](../docs/releases/2026-09-16/candidate_manifest.json)记录了 Full600 启用的功能、关闭的实验、源码哈希与 Prompt/Form 指纹。它是核对依据，不是可直接 source 的环境文件。

默认 `research` profile、`.env.example` 和历史脚本并不自动组合成这份配置；实际运行应检查服务 `/health`。

## 历史基线

`official_nonadk_baseline.json` 与 `official_nonadk_requirements.txt` 属于非 ADK 基线实验，见[历史提交说明](../docs/OFFICIAL_NONADK_GLM47_SUBMISSION.md)。不要用于安装或启动当前 Valibra。
