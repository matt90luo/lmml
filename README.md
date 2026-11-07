# lmml

基于 **Hydra** + **PyTorch Lightning** + **Weights & Biases (`wandb`)** + **Rich** 的 CLI 训练框架（src-layout wheel 包），统一支持 **Qwen-3.5 / MiniMind-3（Dense & MoE）预训练与后训练**、**Kev Decision Model**，以及 **深度强化学习（Rainbow DQN）**。

## 核心特性

1. **LLM 监督微调 (`CausalLMSFTRecipe`)**：
   - 统一管理 **LoRA (`experiment=qwen35_lora_sft`)** 与 **全量微调 (`experiment=qwen35_full_sft`)** 的共享数据配比、ChatML Prompt Masking、模型底座配置与训练超参数。
   - **独立数据准备**：`lmml data` 将 SFT / DPO / PPO 数据源的清洗与分词结果保存为 Google Drive 上的 Arrow `DatasetDict`；训练阶段直接加载，避免重复预处理。
   - **统一 ChatML + CoT + Tools**：兼容 `messages` / `conversations`、`instruction` / `output`、`question` / `answer`、`problem` / `solution` 等源格式；Hermes 的 `tools`、`tool_calls` 与 `tool` 响应会转换为 Qwen ChatML 工具调用模板，已有推理字段会保留在 assistant 目标中。
   - **高吞吐训练管线**：按序列长度分桶采样 (`LengthGroupedSampler`) 及 8 倍数对齐动态 Padding。
   - 训练结束后自动在 CPU 上恢复验证集最优 Checkpoint 并导出 HuggingFace / PEFT 模型与 Tokenizer。

2. **MiniMind-3 / MiniMind-3-MoE 全流程**：
   - 原生 Transformer/MoE 架构（GQA、QK-Norm、RoPE/YaRN、SwiGLU、Top-1 routing）已注册到 Hugging Face AutoClasses。
   - 支持 `pretrain -> full/LoRA SFT -> DPO / PPO / GRPO(CISPO) / Agentic RL`，Dense 与 MoE 使用同一套实验配置。
   - 内置 MiniMind 6400 词表 tokenizer；数据统一为 ChatML + `<think>` CoT + `<tool_call>/<tool_response>`。
   - RL 奖励支持规则/GT、真实 Hugging Face Reward Model 与无外部依赖的 mock 模式。
   - 完整命令见 [`docs/minimind-lmml-migration.md`](docs/minimind-lmml-migration.md)。

3. **Kev Decision Model (`Qwen3.5` + PointerHead)**：
   - 支持 `Noul` / `Choice` / `Score` 强类型判定原语，一次 Prefill 返回概率、置信度与有序分数。
   - 提供 `kev_08b_lora`、`kev_08b_full`、`kev_4b_lora` 与 `kev_08b_smoke` 实验，支持冻结套件预处理、选项增强、`none_pair`、温度校准、CLI/SDK/HTTP 推理。

4. **強化学习 (`RainbowRecipe`)**：
   - 内置完整 6 组件 **Rainbow DQN**（C51 + Dueling + Double DQN + PER + $n$-step + NoisyNet），支持 **纯离线（Offline）**、**纯在线（Online）**、**离线预热转在线（Warmup-then-Online）** 三种训练模式与可插拔 `QNetwork`（MLP / CNN / Transformer）。
5. **WandB 日志与 Rich 终端体验**：
   - 全局默认接入 `WandbLogger` (`project=lmml-projects`)，自动按 `os.environ["WANDB_API_KEY"]` -> `google.colab.userdata.get("WANDB_API_KEY")` -> `~/.netrc` 解析认证。
   - 内置自定义主题 `RichProgressBar` 与 `RichModelSummary(max_depth=2)`。

## Google Colab 快速开始

```python
# 在 Colab 单元格中安装 wheel（按需附加 gpu 算子扩展）并透传 WANDB_API_KEY 至子进程环境变量
import os
from google.colab import userdata

os.environ["WANDB_API_KEY"] = userdata.get("WANDB_API_KEY")
```

```bash
# 1. 准备并分词训练数据（保存到 paths.prepared_data_dir，默认位于 Google Drive）
lmml data experiment=qwen35_lora_sft

# 2. 直接读取已保存数据；LoRA 与 Full SFT 共用同一份预分词结果
lmml train experiment=qwen35_lora_sft
lmml train experiment=qwen35_full_sft

# 3. 覆盖数据源、输出路径或超参数示例
lmml data experiment=qwen35_lora_sft \
  paths.base_data_dir=/content/drive/MyDrive/colab-notes/data/datasets \
  data.prepared_path=/content/drive/MyDrive/colab-notes/data/datasets/prepared/my_sft \
  data.overwrite=true
lmml train experiment=qwen35_lora_sft \
  paths.base_data_dir=/content/drive/MyDrive/colab-notes/data/datasets \
  paths.base_model_dir=/content/drive/MyDrive/colab-notes/data/models \
  algo.lr=2e-4 data.batch_size=8

# 4. Rainbow DQN 训练与离线数据采样
lmml collect env=gridworld collect.n_transitions=20000 collect.policy=uniform collect.save_path=data/grid.npz
lmml train experiment=rainbow_gridworld_online
lmml train experiment=rainbow_gridworld_offline data.offline_path=data/grid.npz
lmml train experiment=rainbow_gridworld_warmup data.offline_path=data/grid.npz data.warmup_offline_steps=1000
```


## Kev Decision Model 快速开始

Kev 是基于 Qwen3.5 的强类型 Decision Model，支持 `Noul` / `Choice` / `Score` 三种判定原语，并在一次 Prefill 中返回概率分布、置信度与有序分数。

```bash
# 1. 准备 Kev 冻结套件（默认读取 Google Drive 的 kev-suits）
lmml data experiment=kev_08b_lora

# 2. 训练：0.8B LoRA（4GB smoke）、0.8B Full、4B LoRA
lmml train experiment=kev_08b_lora
lmml train experiment=kev_08b_full
lmml train experiment=kev_4b_lora

# 3. 预测并查看结构化裁决结果
lmml predict experiment=kev_08b_lora \
  ckpt_path=/content/kev_08b_lora_final \
  predict.output=/content/kev_predictions.jsonl

# 4. 本地 smoke 套件
lmml data experiment=kev_08b_smoke \
  paths.base_model_dir=/home/ukiml/Downloads/models \
  paths.kev_data_dir=/home/ukiml/Downloads/kev/evals
lmml train experiment=kev_08b_smoke
```

训练结束后的温度校准默认按 `algo.calibration.batch_size=1` 分批执行，避免在 4 GB GPU 上一次性前向整个 calibration split。若校准仍然失败，训练流程会导出 `temperature=1.0` 的检查点并把 WandB Run 标记为 Failed，已训练权重不会丢失。

进程内 SDK：

```python
from lmml.core.llm.kev import TypeSafeClient, Choice, Noul, Score

with TypeSafeClient(checkpoint_path="/content/kev_08b_lora_final") as client:
    response = client.system_one(
        state={"text": "上周多扣了我两次费用，马上退款！"},
        questions={
            "refund": Noul(instructions="Does the customer ask for a refund?"),
            "team": Choice(
                instructions="Which team should handle this?",
                criteria={"billing": "Charges", "shipping": "Delivery"},
            ),
            "anger": Score(
                instructions="How angry is the customer?",
                criteria=["calm", "angry"],
            ),
        },
    )
print(response.answers["refund"].noul)
print(response.answers["team"].choice)
print(response.answers["anger"].score)
```

完整数据布局、4GB smoke 参数、HTTP `/v1/systemone` 服务和 TypeSafe 兼容说明见 [`docs/kev-lmml-migration.md`](docs/kev-lmml-migration.md)。

## MiniMind-3 快速开始

```bash
lmml data experiment=minimind3_full_sft
lmml train experiment=minimind3_pretrain
lmml train experiment=minimind3_full_sft
lmml train experiment=minimind3_grpo

# MoE
lmml train experiment=minimind3_moe_pretrain
lmml train experiment=minimind3_moe_full_sft
lmml train experiment=minimind3_moe_agent_rl
```

正式 GPU 训练、Google Drive 数据布局、奖励模型和 4GB smoke test 参数见 `docs/minimind-lmml-migration.md`。
