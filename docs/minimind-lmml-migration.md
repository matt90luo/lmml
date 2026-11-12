# MiniMind-3 / MiniMind-3-MoE 在 lmml 中的训练手册

本手册对应当前 `lmml` 中迁入的 MiniMind 全流程。实现对 MiniMind 参考代码做了两项工程调整：

- 原生模型已经注册为 Hugging Face `AutoConfig` / `AutoModelForCausalLM` 架构（`model_type=minimind`）。
- 训练、评估与导出统一使用 `lmml` 的 Hydra + Lightning + WandB + Rich 生命周期。

## 1. 能力范围

| 阶段 | Dense | MoE | Recipe |
|---|---:|---:|---|
| Pretrain (next-token CE) | 是 | 是 | `causal_lm_sft` |
| Full SFT (assistant-only CE + ChatML/CoT/Tools) | 是 | 是 | `causal_lm_sft` |
| LoRA SFT | 是 | 是 | `causal_lm_sft` |
| DPO | 是 | 是 | `causal_lm_dpo` |
| PPO + Critic + GAE | 是 | 是 | `causal_lm_ppo` |
| GRPO / CISPO | 是 | 是 | `causal_lm_grpo` |
| Agentic RL（最多 3 轮工具交互） | 是 | 是 | `causal_lm_agent_rl` |

所有 MoE 阶段都会将 router 的 auxiliary loss 加入总损失；PPO 的 Actor、Reference、Critic 共享同一 MiniMind 配置。

## 2. 数据布局

默认沿用 Colab 本地权重 `/content` 与 Google Drive 数据：

```text
/content/
├── minimind-3/
└── minimind-3-moe/

/content/drive/MyDrive/colab-notes/data/
└── datasets/minimind_dataset/
    ├── pretrain_t2t_mini.jsonl
    ├── pretrain_t2t.jsonl
    ├── sft_t2t_mini.jsonl
    ├── sft_t2t.jsonl
    ├── dpo.jsonl
    ├── rlaif.jsonl
    ├── agent_rl.jsonl
    └── agent_rl_math.jsonl
```

`minimind-3` / `minimind-3-moe` 目录只需要提供 tokenizer 时，Pretrain 会按代码中的配置随机初始化模型。后续阶段默认从 `paths.base_model_dir`（本地 `/content`）中的 Hugging Face 模型或 `paths.base_dir` 中上一阶段导出的目录加载。

仓库内置了 MiniMind tokenizer 作为离线兜底；当配置路径包含 `minimind` 且该路径不存在时，会使用 `src/lmml/assets/minimind/`。

## 3. 数据预处理

```bash
# Pretrain
lmml data experiment=minimind3_pretrain

# SFT：MiniMind sft_t2t_mini + OpenR1-Math + Hermes Tools
lmml data experiment=minimind3_full_sft

# DPO
lmml data experiment=minimind3_dpo

# PPO / GRPO prompt 数据
lmml data experiment=minimind3_ppo
lmml data experiment=minimind3_grpo

# 多轮 Agent RL
lmml data experiment=minimind3_agent_rl
```

SFT 数据统一输出为下列消息序列，再由 MiniMind tokenizer 的 chat template 渲染：

```text
system(# Tools + <tools>)
user
assistant(<think>...</think> + <tool_call>...</tool_call>)
user(<tool_response>...</tool_response>)
assistant(<think>...</think> + final answer)
```

只有 assistant span 参与 SFT loss；tool response、user 和 padding 均为 `-100`。DPO 分别生成 chosen/rejected 的 assistant mask。RLAIF prompt 保留 `<|im_start|>assistant\n` 与 thinking 起始模板，但不把原答案作为监督目标。

## 4. 完整训练链

```bash
# 1. Dense 预训练
lmml train experiment=minimind3_pretrain

# 2. 全参 SFT（默认读取预训练导出目录，可通过 algo.model.name_or_path 覆盖）
lmml train experiment=minimind3_full_sft

# 3. 后训练分支
lmml train experiment=minimind3_dpo
lmml train experiment=minimind3_ppo
lmml train experiment=minimind3_grpo
lmml train experiment=minimind3_agent_rl
```

MoE 对应实验：

```bash
lmml train experiment=minimind3_moe_pretrain
lmml train experiment=minimind3_moe_full_sft
lmml train experiment=minimind3_moe_lora_sft
lmml train experiment=minimind3_moe_dpo
lmml train experiment=minimind3_moe_ppo
lmml train experiment=minimind3_moe_grpo
lmml train experiment=minimind3_moe_agent_rl
```

显式衔接阶段时覆盖模型目录：

```bash
lmml train experiment=minimind3_ppo \
  algo.model.name_or_path=/content/smoke_sft_final
```

每个 Recipe 在 `on_fit_end` 自动导出 Hugging Face 模型和 tokenizer 到 `algo.export.save_dir`。

## 5. 奖励模型模式

所有 PPO/GRPO/Agent RL 实验支持：

```yaml
algo:
  reward_mode: rule_only   # rule_only | hf | mock
  reward_model_path: null
```

- `rule_only`：长度、thinking 格式、重复 n-gram 与 Agent GT 校验。
- `hf`：在上面规则基础上加载 `InternLM2-1.8B-Reward` 等 Hugging Face Reward Model，原始分数裁剪到 `[-3, 3]`。
- `mock`：轻量测试奖励，用于 4GB 显存或离线 smoke test。

真实奖励模型示例：

```bash
lmml train experiment=minimind3_grpo \
  algo.reward_mode=hf \
  algo.reward_model_path=/content/internlm2-1_8b-reward
```

## 6. 4GB 显存与 CPU 冒烟测试

MiniMind 的 6400 词表显著降低了 logits 显存。推荐的 4GB smoke 参数：

```bash
lmml train experiment=minimind3_grpo \
  data.max_length=128 \
  data.batch_size=1 data.val_batch_size=1 data.num_workers=0 \
  algo.max_new_tokens=16 \
  algo.num_generations=2 \
  trainer.max_epochs=1 trainer.limit_val_batches=0
```

如模型随机初始化且 `max_position_embeddings` 很小，需同时限制 `algo.max_new_tokens`，保证 `prompt + generation <= max_position_embeddings`。

当前开发机器无法通过 NVML 访问显卡（PyTorch 报告 `cuda_available=False`），因此已完成 CPU 微模型全链路验证：

- Dense / MoE Pretrain；
- Full SFT 与 LoRA SFT（含最优 checkpoint 导出）；
- DPO（policy + frozen reference）；
- PPO（actor + critic + GAE，手动优化）；
- GRPO / CISPO（group advantage）；
- Agent RL（tool call、tool response mask、GT reward）；
- RL checkpoint `lmml predict`。

## 7. 验证命令

```bash
.venv/bin/pytest -q
.venv/bin/ruff check src tests
```
