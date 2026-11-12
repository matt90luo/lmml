# Kev Decision Model 在 lmml 中的训练与推理手册

本手册对应 `lmml` 中迁入的 Kev 决策模型全流程。Kev 不是生成式 Chat 模型：它读取一个 `state` 和一组强类型问题，在一次 Prefill 中为每个问题输出概率分布。

支持的判定原语：

| 原语 | 输入 | 输出 |
|---|---|---|
| `Noul` | 是否命题 | `noul` 为 `P(true)` |
| `Choice` | 1..255 个命名选项 | `choice`、`confidence`、`probabilities` |
| `Score` | 有序等级列表 | 期望等级 `score`、`confidence`、`probabilities` |

## 1. 架构

```text
state + <q> instruction <opt> option 0 </opt> ... <decide>
```

- Qwen3.5 的 Gated DeltaNet 是混合线性注意力架构，Kev 使用 **逐问题因果行**：每个问题将 `state + 单个问题分支` 作为独立行前向，分支之间不共享注意力。
- PointerHead 用 `<decide>` 隐状态与每个 `</opt>` 隐状态做缩放点积，得到选项 logits。
- 训练时 Head 的 temperature 固定为 1；推理/评估时使用校准集拟合的 temperature 缩放 logits。
- 用户文本中的 `<|...|>` 会被替换为 `<¦...¦>`，防止伪造选项边界。

## 2. 数据布局

默认路径（基座权重在本地 `/content`，数据在 Google Drive）：

```text
/content/
├── Qwen3.5-0.8B-Base/
└── Qwen3.5-4B-Base/

/content/drive/MyDrive/colab-notes/data/
└── datasets/kev-suits/
    ├── v7/decision-v7/
    ├── documents-v1/
    ├── hard-v1/
    ├── decision-v2/
    └── v3/decision-v3/
```

权重默认放在 `/content`，数据仍从 Google Drive 读取。默认 `kev_suits` 配置会读取 `v7/decision-v7`、`documents-v1`、`hard-v1`；当主套件没有 `train.jsonl` 时，自动回退到 `decision-v2` / `v3/decision-v3` 的 `train.jsonl`。所有记录都会做：

- manifest 的 eval-only / holdout 来源检查；
- 同 split 内 state 去重；
- state / branch 上下文长度准入；
- 输出 Arrow `DatasetDict(train/val/calibration/test)`。

本地仓库可直接指向 `/home/ukiml/Downloads/kev/evals`：

```bash
lmml data experiment=kev_08b_smoke \
  paths.base_model_dir=/home/ukiml/Downloads/models \
  paths.kev_data_dir=/home/ukiml/Downloads/kev/evals \
  paths.prepared_data_dir=/tmp/lmml_kev_data
```

## 3. 实验预设

| 实验 | 基座 | 模式 | 适用场景 |
|---|---|---|---|
| `kev_08b_lora` | Qwen3.5-0.8B-Base | LoRA | 默认，4GB 显存可做 smoke |
| `kev_08b_full` | Qwen3.5-0.8B-Base | Full | 更大显存 |
| `kev_4b_lora` | Qwen3.5-4B-Base | LoRA | 4B 精度 |
| `kev_08b_smoke` | Qwen3.5-0.8B-Base | LoRA | `smoke-v1` 快速验证 |

## 4. 训练

```bash
lmml data experiment=kev_08b_lora
lmml train experiment=kev_08b_lora
```

4GB GPU smoke：

```bash
lmml data experiment=kev_08b_smoke
lmml train experiment=kev_08b_smoke \
  data.batch_size=1 data.val_batch_size=1 data.num_workers=0 \
  algo.calibration.batch_size=1 \
  trainer.max_epochs=1 +trainer.limit_train_batches=4 +trainer.limit_val_batches=1 \
  trainer.accumulate_grad_batches=8 trainer.precision=bf16-mixed
```

训练目标：

- `Choice` / `Noul`：Cross-Entropy；软标签 `Choice` 使用目标分布 CE；
- `Score`：CE + 有序累积分布 RPS 正则；
- 数据增强：选项乱序、`none_of_the_above` 注入、无关干扰项、`none_pair` 最小对比对；
- 训练结束自动在 calibration split 上分批拟合温度（121 点对数网格 `[0.25, 4.0]`），`algo.calibration.batch_size` 默认 `1`；
- 校准前清理 CUDA 缓存，校准完成后先释放 GPU 再导出；若校准失败，仍导出 `temperature=1.0` 的检查点并在 `kev_config.json` 中记录失败原因，同时将 WandB Run 标记为 Failed；
- 导出目录包含 `head.pt`、`adapter_model.safetensors` / 全量权重、tokenizer 和 `kev_config.json`。

## 5. 预测与 SDK

CLI 预测默认使用题述客服工单示例：

```bash
lmml predict experiment=kev_08b_smoke \
  ckpt_path=/content/kev_08b_smoke_final \
  predict.output=/content/kev_predictions.jsonl
```

终端会打印：

```text
Kev decision #1
┌──────────────────────┬─────────┬─────────────────┐
│ Question             │ Result  │ Details         │
├──────────────────────┼─────────┼─────────────────┤
│ asks_for_refund      │ 0.97    │ probability     │
│ routing_queue        │ billing │ confidence=1.00 │
│ customer_frustration │ 2.00    │ confidence=...  │
└──────────────────────┴─────────┴─────────────────┘
```

进程内 SDK 与官方 TypeSafe 调用形状一致：

```python
from lmml.core.llm.kev import TypeSafeClient, Choice, Noul, Score

customer_ticket = {
    "user_id": "U-9402",
    "text": "你们的扣费系统是不是有毛病？上周多扣了我两次费用，马上退款，否则投诉消协！",
    "history_disputes": 0,
}
questions = {
    "asks_for_refund": Noul(instructions="Does the customer explicitly demand a refund of money?"),
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
}

with TypeSafeClient(checkpoint_path="/content/kev_08b_smoke_final", device="cuda") as client:
    response = client.system_one(state=customer_ticket, questions=questions)

print(response.answers["asks_for_refund"].noul)
print(response.answers["routing_queue"].choice)
print(response.answers["routing_queue"].confidence)
print(response.answers["customer_frustration"].score)
```

HTTP 服务（可选依赖）：

```bash
pip install 'lmml[serve]'
python -m lmml.core.llm.kev.serve \
  --checkpoint /content/kev_08b_smoke_final \
  --host 0.0.0.0 --port 8009
```

之后官方 `typesafe_sdk` 可以直接调用：

```python
from typesafe_sdk import TypeSafeClient
client = TypeSafeClient(api_key="local", base_url="http://127.0.0.1:8009", model="kev-latest")
```

HTTP 服务实现 `POST /v1/systemone`、`GET /v1/models`，并返回 `x-typesafe-request-id`。

加载原始 Kev 发布检查点时，如果 `head.pt` 里的 base 是 Hub ID 而本地已有基座，可传 `base_model=/path/to/Qwen3.5-0.8B-Base`；`KevClient`、`KevRuntime` 和 server CLI 都支持这个覆盖。

## 6. 当前环境验证

- 本机 PyTorch 报告 `cuda_available=False`，NVML 不可用，因此无法实际执行 4GB GPU smoke。
- 已完成 CPU 验证：真实 `Qwen3.5-0.8B-Base` 的 hybrid 前向、LoRA 装配、Kev 数据准备、单步训练、分批温度校准、检查点导出、CLI predict 与进程内 SDK。
- 本轮校准修复覆盖分批前向、空校准集、失败时导出 `temperature=1.0` 和 WandB exit code；Ruff：`All checks passed`。
- 本机完整 `pytest` 为 `58 passed`；Ruff 与 `git diff --check` 通过。
- Jev 数据准备已用真实 `jev-distill-corpus-v3` 的小样本完成 CPU 验证：批量预分词、Arrow shard、reservoir sampling、manifest 与完成态复用均通过；真实 4GB GPU smoke 仍待可用 CUDA 主机执行。


## 7. Qwen3.8-27B Jev Decision Model (`qwen38_27B_jev`)

针对 `Qwen3.8-27B` 与 `SargeDev/jev-distill-corpus-v3` 数据集的独立候选打分管线：

- **打分路径**：
  `chat_template(context, question, candidate) -> Qwen3.8-27B language_model + LoRA -> last non-padding token hidden state -> FP32 Linear(hidden_size, 1) -> candidate scalar score -> softmax over candidates`
- **两段式加载 (`QwenLoraLightningModule`)**：
  1. 冻结 `Qwen3.8-27B` 纯文本主干 `language_model`，并用预训练输出层 `"Yes" - "No"` 行向量差初始化 `FP32 Linear(hidden_size, 1)`；
  2. 若指定 `algo.lora_checkpoint_dir` 且目录存在，通过 `PeftModel.from_pretrained(..., is_trainable=True)` 恢复 LoRA 权重及 `head.pt` / `temperature.json`；否则按 `LoraConfig(r=32, lora_alpha=64)` 全新初始化。
- **损失函数**：软目标交叉熵 + `0.1 * Brier Loss`。
- **数据准备优化**：
  - 按 `shard_size=20000` 行分片，worker 使用 `Dataset.from_generator` 流式写 Arrow，避免 65 万行一次性进入 Python list；
  - 按 `tokenization_batch_size=512` 条记录批量渲染候选并 tokenize，保存 `input_ids_flat` / `candidate_offsets` / `candidate_lengths`，训练 collator 只做 padding；
  - `max_samples` 使用全文件 reservoir sampling；训练 split 默认按 `(source, kind)` 保持 round-robin 平衡；
  - `preprocessing_num_proc=auto` 最多 8 进程，每 10,000 行打印进度；
  - staging shard 完成后写入 `.done.json`，同一配置重跑自动复用已完成 shard；源文件或 tokenizer 变化时拒绝错误复用；
  - 正式配置使用完整 validation / calibration / test / test_set_30k / ood；smoke 配置单独限制小样本；
  - 新产物不再保存重复的 `record_json`，而是保存结构化列和预分词字段；旧 `record_json` 数据仍可由 collator 兼容读取。
- **三阶段流程**：
  - **阶段一（训练）**：在 `train.jsonl` 上更新，在 `validation.jsonl` 上监控 `val/loss`，自动将最优 Top-2 检查点保存在 `${paths.output_model_dir}/checkpoints_qwen38_27b_jev`（含轻量 `.ckpt` 与 `top1_*` / `top2_*` PEFT 目录）；
  - **阶段二（校准）**：在 `calibration.jsonl` 上拟合最优正温度系数 `T` 以压低 ECE，写入 `temperature.json`；
  - **阶段三（验收）**：通过 `lmml test experiment=qwen38_27B_jev`（或 `eval.after_train=true`）依次评测 `test.jsonl`（功能与吞吐）、`test_set_30k.jsonl`（Jev Decision Index 指标）与 `ood.jsonl`（System 1/2 升级触发率与抗盲目自信指标）。
