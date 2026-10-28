# lmml

基于 **Hydra** + **PyTorch Lightning** + **Weights & Biases (`wandb`)** + **Rich** 的 CLI 训练框架（src-layout wheel 包），统一支持 **大语言模型监督微调（Qwen-3.5 LoRA / Full SFT）** 与 **深度强化学习（Rainbow DQN）**。

## 核心特性

1. **LLM 监督微调 (`CausalLMSFTRecipe`)**：
   - 统一管理 **LoRA (`experiment=qwen35_lora_sft`)** 与 **全量微调 (`experiment=qwen35_full_sft`)** 的共享数据配比、ChatML Prompt Masking、模型底座配置与训练超参数。
   - **独立数据准备**：`lmml data` 将 SFT / DPO / PPO 数据源的清洗与分词结果保存为 Google Drive 上的 Arrow `DatasetDict`；训练阶段直接加载，避免重复预处理。
   - **高吞吐训练管线**：按序列长度分桶采样 (`LengthGroupedSampler`) 及 8 倍数对齐动态 Padding。
   - 训练结束后自动在 CPU 上恢复验证集最优 Checkpoint 并导出 HuggingFace / PEFT 模型与 Tokenizer。
2. **強化学习 (`RainbowRecipe`)**：
   - 内置完整 6 组件 **Rainbow DQN**（C51 + Dueling + Double DQN + PER + $n$-step + NoisyNet），支持 **纯离线（Offline）**、**纯在线（Online）**、**离线预热转在线（Warmup-then-Online）** 三种训练模式与可插拔 `QNetwork`（MLP / CNN / Transformer）。
3. **WandB 日志与 Rich 终端体验**：
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
