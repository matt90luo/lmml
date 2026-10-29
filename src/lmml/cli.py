"""lmml CLI 入口：支持子命令 data / train / test / predict / collect + Hydra 配置与参数覆盖。"""

from __future__ import annotations

import sys

import hydra
from omegaconf import DictConfig

from lmml.__about__ import __version__

COMMANDS = ("data", "train", "test", "predict", "collect")

USAGE = f"""lmml v{__version__} - RL / LLM / ML / DL training CLI (Hydra + Lightning + WandB + Rich)

Usage:
  lmml <command> [hydra_overrides...]

Commands:
  data      Prepare and tokenize LLM datasets for later training
  train     Train an RL or LLM SFT model (LoRA, Full SFT, online/offline/warmup RL)
  test      Evaluate a trained model or checkpoint
  predict   Run inference / generation and export predictions
  collect   Collect offline transitions from a Gymnasium environment

Examples:
  # Prepare LLM datasets once, then train directly from Google Drive
  lmml data experiment=qwen35_lora_sft
  lmml train experiment=qwen35_lora_sft

  # LLM SFT (Qwen-3.5 0.8B LoRA & Full Fine-Tuning on Colab)
  lmml train experiment=qwen35_lora_sft
  lmml train experiment=qwen35_full_sft
  lmml train experiment=qwen35_lora_sft paths.base_data_dir=/content/data trainer.max_epochs=1

  # MiniMind-3 Dense / MoE (pretrain, SFT, DPO, PPO, GRPO/CISPO, Agent RL)
  lmml data experiment=minimind3_full_sft
  lmml train experiment=minimind3_pretrain
  lmml train experiment=minimind3_full_sft
  lmml train experiment=minimind3_grpo
  lmml train experiment=minimind3_moe_agent_rl

  # RL (Rainbow DQN)
  lmml train experiment=rainbow_gridworld_online
  lmml train data=offline data.offline_path=data/grid.npz net=transformer
  lmml train data=warmup_online data.offline_path=data/grid.npz data.warmup_offline_steps=1000
  lmml test ckpt_path=outputs/last.ckpt
  lmml predict ckpt_path=outputs/last.ckpt predict.output=preds.npz
  lmml collect env=gridworld collect.n_transitions=10000 collect.policy=uniform collect.save_path=data/grid.npz
"""


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def _hydra_entry(cfg: DictConfig) -> float | None:
    from lmml.core.runner import run

    res = run(cfg)
    return res if isinstance(res, (int, float)) else None


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help", "help"):
        print(USAGE)
        return
    if args[0] in ("-V", "--version", "version"):
        print(__version__)
        return
    if args[0] in COMMANDS:
        cmd, rest = args[0], args[1:]
        if any(a in ("-h", "--help") for a in rest):
            print(USAGE)
            return
        sys.argv = [sys.argv[0], f"command={cmd}", *rest]
    elif not any(a.startswith("command=") for a in args):
        sys.exit(f"Unknown subcommand {args[0]!r}. Choose from {COMMANDS} or pass --help.")
    else:
        sys.argv = [sys.argv[0], *args]
    _hydra_entry()


if __name__ == "__main__":
    main()
