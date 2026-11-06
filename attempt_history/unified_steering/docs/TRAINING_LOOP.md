# M2：网络与recurrent PPO闭环

状态：本文记录已实现的直接关节 PPO 与历史运行，不再安排主线训练。旧检查点、日志和详细报告已按用户要求清理，下文命令仅作接口参考，不能直接恢复旧运行。腕手教师、历史学生及其阶段门槛统一见[最新研究计划](RESEARCH_PLAN.md)。

本阶段将M1环境接入真实梯度训练。M2的工程预算是5万–10万交互步，不要求在该预算内学会方向盘任务。后续历史M3试训也未通过可靠跟踪。固定开发测试结果与训练奖励分开报告。

seed7曾完成51,200步，工程检查通过，固定开发精度未通过；结论见[清理后的历史摘要](../results/README.md)。

M3-0的定期固定评价、按步数保留checkpoint和冻结探索对照曾实现并验证，接口见[使用说明](DIAGNOSTICS.md)，相关原始产物已清理。原M3后续路线已被最新计划取代；下文保留M2已执行设置。

## 网络与信息边界

Actor共258,740个参数，critic共252,929个参数。网络与M1接口对应：

```text
152维部署观测
  ├─除参考以外140维 → Linear256/ELU → Linear128/ELU → GRU128 ─┐
  └─11点参考误差+当前参考速度12维 → Linear64/ELU → Linear64/ELU ─┴─192
      → Linear256/ELU → Linear128/ELU → Linear26
      → 26维tanh Gaussian

172维critic观测 → Linear512/ELU → Linear256/ELU → Linear128/ELU → Linear1
```

Actor与critic没有共享层，critic特权数据没有进入actor接口。所有actor层、GRU和26维log标准差一起优化。臂初始确定性均值接近零速度，手均值接近已有抓握；并非26维全零初始化。线性层正交初始化，输出权重增益0.01。

M1已经固定各字段量纲，本阶段不做在线观测/奖励归一化。checkpoint明确保存该缩放契约；没有需要恢复的运行均值或方差。

## 动作概率

先采样未压缩潜变量`z ~ Normal(mean,std)`，环境输入为`a=tanh(z)`。计算概率使用：

```text
log π(a) = Σ[log Normal(z) − log(1−tanh(z)²)]
log(1−tanh(z)²) = 2[log 2 − z − softplus(−2z)]
```

保存z、网络采样动作a、旧log概率及执行器实际动作。PPO用z重算采样动作概率，不能把执行后受限的关节速度/位置当作原始采样动作。保存z也避免对饱和tanh输出做不稳定的atanh。训练中log标准差限制在[-5,1]，初始臂/手潜变量标准差分别0.35/0.50。

熵使用当前压缩分布的一次重参数化采样估计，系数0.001；它是连续分布的微分熵，可以为负。

## 时序采集和GRU

`workers.py`在独立进程运行MuJoCo，每个worker一个环境。接口分别返回动作后最终观测、下一次采样观测以及结束标记。自动reset时二者明确分开，不使用reset后的critic观测给上一回合bootstrap。

Rollout保持`[时间,环境,字段]`顺序，并保存episode开始标记。M2用4环境×128步采集，每64步为一段，按整段组成256步minibatch，不把序列打散成单帧。段内发生reset时，在该帧前清空GRU状态，也切断跨回合梯度。

Collector保留每个正在运行回合的部署观测历史。每轮更新后，以最新actor权重从该回合起点重算记忆，避免沿用旧权重产生的隐状态。每个训练片段也从最近reset重算历史：片段前的历史不传播梯度，64步片段内进行反向传播。历史长度在当前15秒回合内有界；后续扩大回合和环境数时需要重新测量该方法的开销。

每轮PPO更新前，冻结当前权重对全部序列重算旧动作概率。概率比应接近1，最大偏差超过1e−4立即停止；实际误差另行记录，单元检查使用更紧的1e−5门槛。

## GAE和优化

每个采集步保存动作后真实最终状态的旧critic估值。公式为：

```text
δ_t = r_t + γ(1−terminated_t)V(final_state_t) − V(state_t)
A_t = δ_t + γλ[1−(terminated_t or truncated_t)]A_(t+1)
return_t = A_t + V(state_t)
```

真实失败不bootstrap。时间截断使用真实最终状态bootstrap，但优势递推不穿过reset。未结束的rollout末尾也使用最后状态的价值。折扣按控制决策步计算。

PPO使用clip=0.2的最小截断代理目标，整批rollout优势归一化；critic损失为0.5倍均方误差，不裁剪价值。Actor和critic分别使用Adam，学习率1e−4/3e−4，epsilon=1e−5，各自梯度范数限制1。γ=0.995、λ=0.97，每轮最多4个epoch。近似KL用`mean(exp(log_ratio)−1−log_ratio)`，大于1.5×0.015时停止当轮后续更新。

每轮记录KL、clip比例、熵、价值损失、解释方差、裁剪前梯度范数、标准差、概率重算误差、环境奖励分项和执行层介入。损失、梯度、参数及采集数据必须有限；环境数值失效会停止训练，不能把回退观测继续作为有效数据学习。

## 算法来源

本实现参考并检查了本机Stable-Baselines3 **2.6.0**的GAE、clipped PPO目标及KL提前停止逻辑；GRU、独立critic、显式最终观测及历史重算由本项目实现，并非直接调用SB3或sb3-contrib的recurrent trainer。SB3不是本项目运行依赖。

- [SB3 PPO源码，v2.6.0](https://github.com/DLR-RM/stable-baselines3/blob/v2.6.0/stable_baselines3/ppo/ppo.py)，本机文件SHA256：`f29c1286decfbd0d15341dd3f41f876cedd7069428abab5fe7cfebfbeb487147`。
- [SB3 rollout buffer，v2.6.0](https://github.com/DLR-RM/stable-baselines3/blob/v2.6.0/stable_baselines3/common/buffers.py)，SHA256：`973c1e891f42b5aac185385e9b3b845eb3c78843d95e355cb55072aa316f8144`。
- SB3作者版权及[MIT许可证](https://github.com/DLR-RM/stable-baselines3/blob/v2.6.0/LICENSE)；本项目重新实现相应算法公式，没有复制整个trainer。
- [PPO](https://arxiv.org/abs/1707.06347)、[GAE](https://arxiv.org/abs/1506.02438)；tanh概率修正通过PyTorch `TransformedDistribution`独立核对。

## M2运行设置

`configs/m2_smoke.yaml`继承候选网络与PPO参数，覆盖为4个CPU环境、seed7、51,200步、128步rollout、64步序列、256步minibatch。使用已抓握双手初态，保留全26维学习动作和名义回正负载。

25%回合保持初始角，75%回合跟踪随机选择先正或先负的±5°往返，参考速度5°/s，回合15秒。每次reset只选择任务，运行中没有接管调度。M2尚未实现完整幅度课程、单手课程或动力学随机化；`stage_a.yaml`中后续课程条目属于M3及以后设计，不代表本次已执行。

固定开发评价包含零附近保持、正负5°偏置目标、两个方向的5°往返。对比训练前确定性actor、训练后确定性actor和固定关节目标。记录角度RMSE、峰值、末段3秒漂移/目标误差、无硬失败完成率、跟踪成功率与执行介入。提前失败的短轨迹RMSE不能与完整轨迹RMSE直接解释为优劣，必须一起看完成情况。这是开发测试，不是未见轨迹泛化或论文统计。

## 保存、恢复和命令

Checkpoint保存网络、两个优化器、完整配置、接口/固定量纲契约、步数以及Python/NumPy/Torch RNG。恢复后仿真环境从新回合开始、GRU清零；不声称无缝恢复到同一个物理子步。仅加载本项目生成、可信的本地checkpoint。

```bash
bash scripts/run.sh -m unittest discover -s tests -v
bash scripts/run.sh -m unified_steering.train --output runs/m2_smoke_new_run

# 独立确定性评价或固定关节目标对照
bash scripts/run.sh -m unified_steering.evaluate \
  --checkpoint runs/m2_smoke_new_run/final.pt --output outputs/m2_eval_new_run
bash scripts/run.sh -m unified_steering.evaluate \
  --checkpoint runs/m2_smoke_new_run/final.pt --fixed-targets --output outputs/m2_fixed_new_run

# 恢复到新的输出目录；total-steps为包含已有步数的总预算
bash scripts/run.sh -m unified_steering.train \
  --resume runs/m2_smoke_new_run/final.pt --total-steps 102400 --output runs/m2_resumed_new_run
```

输出包含`resolved_config`、代码/模型摘要、初始及最终checkpoint、逐轮CSV、逐回合JSONL和固定开发评价CSV。每个输出目录拒绝覆盖。

普通MuJoCo的物理在CPU，learner可以配置为CPU或CUDA；GPU批量物理需要MJX等后端的单独迁移。当前CPU/CUDA短基准可复现：

```bash
bash scripts/run.sh -m unified_steering.benchmark_learning --output outputs/m2_cpu_cuda_new_run
```

该命令以相同4环境配置分别预热并测量采集+PPO更新，报告同步后的CUDA计时。它不是GPU物理仿真或完整CUDA训练结果；更大批量的性能需要重新测量。
