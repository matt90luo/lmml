# lmml

基于 **Hydra** + **PyTorch Lightning** + **TensorBoard** 的 CLI 训练框架（src-layout wheel 包），支持强化学习与通用机器学习/深度学习模型。

首个内置算法为 **Rainbow DQN**（C51 + Dueling + Double DQN + PER + $n$-step + NoisyNet），支持 **纯离线（Offline）**、**纯在线（Online）**、**离线预热转在线（Warmup-then-Online，方案 A）** 三种模式，以及可插拔的 `QNetwork` 结构（MLP / CNN / Transformer）。

## 快速使用

```bash
# 1. 采样棋盘环境离线数据（默认 uniform 策略）
lmml collect env=gridworld collect.n_transitions=20000 collect.policy=uniform collect.save_path=data/grid.npz

# 2. 纯在线训练 Rainbow DQN
lmml train experiment=rainbow_gridworld_online

# 3. 纯离线训练（训练循环完全不调用 env.step）
lmml train experiment=rainbow_gridworld_offline data.offline_path=data/grid.npz

# 4. 方案 A：离线预热 + 在线微调
lmml train experiment=rainbow_gridworld_warmup data.offline_path=data/grid.npz data.warmup_offline_steps=1000

# 5. 切换不同 Q(s,a) 网络结构（MLP / Transformer / CNN）
lmml train experiment=rainbow_gridworld_online net=transformer
lmml train experiment=rainbow_gridworld_online env.kwargs.obs_mode=image net=cnn

# 6. 测试与预测导出
lmml test ckpt_path=outputs/.../last.ckpt
lmml predict ckpt_path=outputs/.../last.ckpt predict.output=predictions.npz
```
