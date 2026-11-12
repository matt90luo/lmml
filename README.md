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

3. **Jev Decision Model (`Qwen3.8-27B` + Rank-32 LoRA + FP32 Scalar Head)**：
   - 对齐 Open-Jev 独立候选打分架构：`chat_template(context, question, candidate) -> Qwen3.8-27B language_model + LoRA -> last non-padding token hidden state -> FP32 Linear(hidden_size, 1) -> softmax`。
   - 支持两段式加载（冻结基座 + 新建或从 `lora_checkpoint_dir` 恢复 LoRA & Head 权重）、软目标 `Cross-Entropy + 0.1 * Brier Loss`、Validation `save_top_k=2` 保存至 Google Drive、阶段二温度校准与阶段三（`test` / `test_set_30k` / `ood`）验收。
   - 默认使用 `AdamW + 5% Warmup + Cosine Decay`、训练期分层采样、按优化器步数触发验证、FlashAttention-2 自动回退与 Liger RMSNorm/SwiGLU 融合；附带 FakeTensor 显存峰值估算工具。

4. **Kev Decision Model (`Qwen3.5` + PointerHead)**：
   - 支持 `Noul` / `Choice` / `Score` 强类型判定原语，一次 Prefill 返回概率、置信度与有序分数。
   - 提供 `kev_08b_lora`、`kev_08b_full`、`kev_4b_lora` 与 `kev_08b_smoke` 实验，支持冻结套件预处理、选项增强、`none_pair`、温度校准、CLI/SDK/HTTP 推理。

5. **強化学习 (`RainbowRecipe`)**：
   - 内置完整 6 组件 **Rainbow DQN**（C51 + Dueling + Double DQN + PER + $n$-step + NoisyNet），支持 **纯离线（Offline）**、**纯在线（Online）**、**离线预热转在线（Warmup-then-Online）** 三种训练模式与可插拔 `QNetwork`（MLP / CNN / Transformer）。
6. **WandB 日志与 Rich 终端体验**：
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
  paths.base_model_dir=/content \
  algo.lr=2e-4 data.batch_size=8

# 4. Rainbow DQN 训练与离线数据采样
lmml collect env=gridworld collect.n_transitions=20000 collect.policy=uniform collect.save_path=data/grid.npz
lmml train experiment=rainbow_gridworld_online
lmml train experiment=rainbow_gridworld_offline data.offline_path=data/grid.npz
lmml train experiment=rainbow_gridworld_warmup data.offline_path=data/grid.npz data.warmup_offline_steps=1000
```


## Qwen3.8-27B Jev Decision Model (`qwen38_27B_jev`) 快速开始

基于 `Qwen3.8-27B` (`Qwen3_5ForConditionalGeneration` 提取 `language_model`) 与 `SargeDev/jev-distill-corpus-v3` 数据集的独立候选打分型 Decision Model：

```bash
# 0. 权重下载到本地 /content（比直接读取 Google Drive 挂载目录更快）
hf download Qwen/Qwen3.8-27B --local-dir /content/Qwen3.8-27B

# 1. 准备数据（读取 Google Drive 上的 jev-distill-corpus-v3 六个 split）
#    默认流式分片 + 批量预分词 + 最多 8 进程；中断后重复执行会复用已完成 shard
lmml data experiment=qwen38_27B_jev

# 小样本/快速验证使用 smoke 配置（单进程、小 shard）
lmml data experiment=qwen38_27B_jev_smoke

# 2. 阶段一训练（Rank-32 LoRA + CE + 0.1*Brier Loss，保存 Validation Top-2 至 Google Drive）
#    训练结束自动执行阶段二 calibration.jsonl 温度校准并导出最终检查点
lmml train experiment=qwen38_27B_jev

# 从已有 LoRA Checkpoint 恢复继续训练（两段式加载 Case A）
lmml train experiment=qwen38_27B_jev \
  algo.lora_checkpoint_dir=/content/drive/MyDrive/colab-notes/data/models/qwen38_27b_jev_final

# 3. 阶段三最终验收（依次评测 test / test_set_30k / ood 并输出 evaluation_summary.json）
lmml test experiment=qwen38_27B_jev \
  ckpt_path=/content/drive/MyDrive/colab-notes/data/models/qwen38_27b_jev_final

# 4. 预测并输出客服工单 Noul / Choice / Score 结构化裁决结果
lmml predict experiment=qwen38_27B_jev \
  ckpt_path=/content/drive/MyDrive/colab-notes/data/models/qwen38_27b_jev_final \
  predict.output=/content/jev_predictions.jsonl

# 5. 训练期采样：默认 64,000 train / 1,024 validation，无需重跑 lmml data
#    如需全量 643,500 条训练集，直接覆盖为 null
lmml train experiment=qwen38_27B_jev data.train_max_samples=null
lmml train experiment=qwen38_27B_jev data.train_max_samples=128000 data.val_max_samples=2048
```

### Batch、Step 与 WandB 对应关系

默认训练参数为 `data.batch_size=2`、`trainer.accumulate_grad_batches=16`，单卡等效 `global_batch_size=32`。一个 64,000 条样本的 Epoch 包含 32,000 个 micro-batch、2,000 次优化器更新：

| 指标 | 含义 | 默认值 |
| :--- | :--- | :--- |
| `micro_batch_size` | 每次 forward/backward 的 question 行数；每个 question 再展开候选序列 | 2 |
| `accumulate_grad_batches` | 累积多少个 micro-batch 才执行一次 `optimizer.step()` | 16 |
| `global_batch_size` | `micro_batch_size * accumulate_grad_batches * devices` | 32 |
| `micro_batches/epoch` | 进度条与 Lightning 的 `batch_idx` 范围 | 32,000 |
| `optimizer_steps/epoch` | WandB 横轴 `trainer/global_step` 的总步数 | 2,000 |
| `val_every_n_opt_steps` | 每多少次优化器更新执行一次中途验证 | 250 |
| `val_check_interval` | Lightning 实际使用的 micro-batch 验证间隔 | 4,000 |

WandB 中重点查看 `train/loss`、`train/ce_loss`、`train/brier_loss`、`val/loss`、`val/expected_acc`、`val/ece` 以及 `lr-AdamW/pg*`。验证间隔由 `algo.val_every_n_opt_steps` 自动换算，不再手动把优化器步数误写成 dataloader batch 数。

### Liger-Kernel、FlashAttention-2 与显存估算

- `algo.model.attn_implementation=flash_attention_2`：Colab 安装 `flash-attn` 且 CUDA 可用时启用；否则自动回退到 `sdpa`，不影响训练。
- `algo.model.use_liger_kernel=true`：Jev 模型只融合 `Qwen3.5/3.8` 的 RMSNorm 与 SwiGLU MLP，不启用 FLCE（因为 Jev 已剥离 `lm_head`）；标准 CausalLM SFT 可同时启用 FLCE 消除词表 Logits 峰值。缺少 Liger 或运行在 CPU 时自动回退。
- 本地无 GPU 也可以先估算显存：

```bash
python -m lmml.utils.vram_estimator \
  --model /content/Qwen3.8-27B \
  --mode jev --seq-len 512 --candidates 4 \
  --vram-gb 80 --max-batch 32

python -m lmml.utils.vram_estimator \
  --model /content/Qwen3.8-27B \
  --mode causal_lm --seq-len 2048 --vram-gb 80
```

估算器使用 `FakeTensorMode` 执行一次 forward/backward/AdamW step，并统计 fake storage 的峰值字节数；不会占用真实 GPU 显存。

进程内 `TypeSafeClient` 调用：

```python
from lmml.core.llm.jev import TypeSafeClient, Choice, Noul, Score

with TypeSafeClient(
    checkpoint_path="/content/drive/MyDrive/colab-notes/data/models/qwen38_27b_jev_final"
) as client:
    response = client.system_one(
        state={
            "user_id": "U-9402",
            "text": "你们的扣费系统是不是有毛病？上周多扣了我两次费用，马上退款，否则投诉消协！",
            "history_disputes": 0,
        },
        questions={
            "asks_for_refund": Noul(
                instructions="Does the customer explicitly demand a refund of money?"
            ),
            "routing_queue": Choice(
                instructions="Which department should handle this request?",
                criteria={
                    "billing": "Issues with recurring payments, charges, cards, or receipts",
                    "account_security": "Compromised accounts, password resets, 2FA failures",
                    "technical_support": "App crashes, 5xx errors, API bugs",
                    "general_inquiry": "Other non-urgent questions",
                },
            ),
            "customer_frustration": Score(
                instructions="Rate the level of user emotional frustration and hostility.",
                criteria=[
                    "Level 0: Calm, objective, neutral tone",
                    "Level 1: Annoyed, mildly impatient but polite",
                    "Level 2: Openly aggressive, angry, threatening legal or external escalations",
                ],
            ),
        },
    )
print(response.answers["asks_for_refund"].noul)
print(response.answers["routing_queue"].choice)
print(response.answers["customer_frustration"].score)
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

Jev 正式数据准备的默认参数为 `preprocessing_num_proc=auto`（最多 8）、`tokenization_batch_size=512`、`shard_size=20000`、`progress_every=10000`。预处理产物保存预分词候选 `input_ids_flat` / `candidate_offsets` / `candidate_lengths`，训练时只做 padding，不再每个 epoch 重新 tokenize。详细配置见 `src/lmml/configs/data/jev_distill_v3.yaml`。

基座权重默认从本地 `/content/<model>` 加载；训练检查点与最终导出仍保存到 `${paths.output_model_dir}`（默认 Google Drive）。可通过 `paths.base_model_dir` 覆盖基座目录，通过 `paths.output_model_dir` 覆盖产物目录。

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
