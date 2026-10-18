# 迁移指南：从 PD 位置控制到残差力矩控制

本文档记录了将 `go2_jump` 任务从一个**模拟 PD 位置控制（Position Control）策略**迁移到一个**残差力矩控制（Residual Torque Control）**策略所做的核心代码修改。

## 1. 核心概念转变

### 原 PD 位置控制 (Before)
在原始版本中，策略网络 (Policy) 输出的是一个目标关节位置（actions）。  
`control.action_scale` 被设置为一个很小的值（例如 0.25），代表弧度。  
`_compute_torques` 函数在软件中实现了一个 PD 控制器，用于计算跟踪这个目标位置所需的力矩。

### 残差力矩控制 (After)
在修改后的版本中，策略网络输出的是一个残差力矩（Residual Torque）。  
PD 控制器仍然存在，但其角色变为一个基础稳定器，始终试图将机器人拉回其默认姿态（`default_dof_pos`）。  
策略网络学习输出一个“额外”的力矩（actions），用于执行跳跃等动态动作。  
`_compute_torques` 函数将“基础 PD 力矩”和“策略残差力矩”相加，得到最终发送给电机的总力矩。

## 2. 关键代码修改

### 步骤 1：修改 `GO2_JUMP_config.py`
- **更改 `control.action_scale` (最关键)**:  
  原代码：  
  ```python
  action_scale = 0.25  # 代表 0.25 弧度
  ```  
  修改后:  
  ```python
  action_scale = 10.0  # 代表 10.0 牛顿·米 (Nm)
  ```

- **更改 `asset.file`**:  
  为了确保仿真器正确处理力矩，将资源文件指向一个明确定义了 effort 驱动模式的 URDF。  
  原代码：  
  ```python
  file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf'
  ```  
  修改后 (参考 sata 库)：  
  ```python
  file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2_torque.urdf'
  ```

### 步骤 2：修改 `go2_jump_env.py`
- **重写 `_compute_torques` 函数**:  
  原代码 (PD 位置跟踪):  
  ```python
  def _compute_torques(self, actions):
      # ...
      p_gains = self.p_gains * self.p_gains_multiplier
      d_gains = self.d_gains * self.d_gains_multiplier
      # 'actions' 是缩放后的目标位置偏移
      torques = p_gains * (actions + self.default_dof_pos - self.dof_pos + self.motor_zero_offsets) - d_gains * self.dof_vel
      return torch.clip(torques, -self.torque_limits, self.torque_limits)
  ```  
  修改后 (残差力矩):  
  ```python
  def _compute_torques(self, actions):
      # ...
      p_gains = self.p_gains * self.p_gains_multiplier
      d_gains = self.d_gains * self.d_gains_multiplier
      # 1. 计算基础 PD 力矩 (用于稳定在默认姿态)
      pd_torques = p_gains * (self.default_dof_pos - self.dof_pos + self.motor_zero_offsets) - d_gains * self.dof_vel
      # 'actions' 是缩放后的残差力矩
      # 2. 叠加策略网络输出的残差力矩
      torques = pd_torques + actions
      return torch.clip(torques, -self.torque_limits, self.torque_limits)
  ```

- **修改 `_reward_torques` 函数 (逻辑变更)**:  
  原代码:  
  ```python
  def _reward_torques(self):
      # 惩罚最终的总力矩
      return torch.sum(torch.abs(self.torques), dim=1)
  ```  
  修改后 (参考 sata):  
  ```python
  def _reward_torques(self):
      # 惩罚策略网络输出的“残差力矩”（self.actions 是未缩放的策略输出）
      # 这鼓励策略“偷懒”，多依赖基础 PD 控制器
      return torch.sum(torch.square(self.actions), dim=1)
  ```

## 3. 结论
完成以上修改后，策略网络的输出 `actions` 的物理含义从“目标位置”（弧度）彻底转变为“残差力矩”（牛顿·米）。
```python
python legged_gym/scripts/play.py --task=go2_jump
  ```  

## 4. Sim2Sim 部署与对比测试 (Mujoco)

本项目提供两套独立的部署脚本，分别用于在相同的 Mujoco 物理环境中测试 **力矩控制（Torque Control）** 和 **位置控制（Position Control）** 两种策略，便于直观对比两种控制方式在爆发力、抗干扰性上的差异。

### 4.1 前置步骤：导出 JIT 模型（TorchScript）

`sim2sim` 脚本只能加载 TorchScript（JIT）格式的模型。运行仿真前，必须先使用 `play.py` 将训练好的 `.pt` 权重导出为 JIT 模型。

#### 1. 导出力矩控制策略（Torque Policy）
```bash
python legged_gym/scripts/play.py --task go2_jump --load_run <你的RunName> --checkpoint <你的模型编号>
```
导出后模型位于：  
`logs/go2_jump/exported/policies/policy_*.pt`（推荐使用最新的 `policy_1.pt`）

#### 2. 导出位置控制策略（Position Policy）
```bash
python legged_gym/scripts/play.py --task go2_jump_pos --load_run <你的RunName> --checkpoint <你的模型编号>
```
导出后模型位于：  
`logs/go2_jump_pos/exported/policies/policy_*.pt`

### 4.2 运行 Mujoco 仿真对比测试

#### A. 力矩控制模式（Torque Control）
对齐 Isaac Gym 训练环境，使用动态 Action Scale（Iter 4000 时为 23.5Nm），实现纯力矩部署。
```bash
python3 deploy_mujoco/sim2sim_GO2_jump_torque.py --load_model logs/go2_jump_torque/exported/policies/policy_1.pt
```

#### B. 位置控制模式（Position Control）
使用较高的 PD 参数（Kp=20.0, Kd=0.5），网络输出被解析为目标关节角（Action Scale=0.25），更加平滑稳定。
```bash
python3 deploy_mujoco/sim2sim_GO2_jump_position.py --load_model logs/go2_jump_pos/exported/policies/policy_1.pt
```

### 4.3 键盘控制说明（仿真窗口获得焦点后生效）

| 按键      | 功能                  | 指令变化                  |
|-----------|-----------------------|---------------------------|
| `W` / `S` | 前进 / 后退           | x_vel_cmd                 |
| `A` / `D` | 左移 / 右移           | y_vel_cmd                 |
| `Q` / `E` | 左转 / 右转           | yaw_vel_cmd               |
| `J`       | **爆发跳跃 (Surge)**  | 触发 1.0m/s 瞬时前冲指令  |
| `I` / `K` | 加速 / 减速           | 调整全局速度倍率          |

> Tips：力矩控制模式下，按下 `J` 键可观察到机器人在纯力矩驱动下的爆发跳跃表现。指令响应已调优，建议先按 `W` 让机器人进入运动状态后再尝试跳跃。

## 5. 项目进展与演化 (Project Progress & Evolution)

### 5.1 解决“小腿不自然上抬”问题 (Solving "Unnatural Calf Tucking")
在训练的纯力矩阶段，观察到机器人为了“作弊”获取 `feet_clearance` 奖励，学会了不自然地向上收缩（tuck）小腿，而不是通过驱动整个身体来跳跃。

**解决方案**: 通过动态增强对小腿关节（calf joints）的姿态惩罚（`_reward_default_pos`），我们成功地抑制了这种行为。当PD辅助控制器衰减后，对小腿姿态偏离的惩罚会变得更重，从而迫使AI学习使用力矩来维持正确的腿部姿态。

### 5.2 速度极限挑战与性能对标 (Velocity Challenge & Benchmarking)

**核心基准模型 (Core Benchmark Model)**:
- **Run Name**: `Mar05_07-28-45_` (训练日期: 2026-03-05)
- **Checkpoint**: `model_3500.pt` (3500 代迭代模型)
- **控制方式**: 纯力矩控制 (Residual Torque, PD Factor=0, Action Scale=23.5 Nm)

**环境精度表现 (Steady-state Accuracy)**:
- **Isaac Gym (理想环境)**: 稳定跟踪 **4.0 m/s** 极速。
- **Mujoco (Sim2Sim Final)**: 稳定跟踪 **2.5 m/s** (高保真物理对齐版)。
- **鲁棒性验证**: 成功通过了 **5.0 kg - 8.0 kg (超载 70%+)** 额外负载及**崎岖地形 (--terrain)** 的压力测试。

**鲁棒性压力测试 (Robustness Stress Test)**:
为了对两种控制策略进行公平、统一的鲁棒性对比，两个部署脚本 (`sim2sim_GO2_jump_torque_final.py` 和 `sim2sim_GO2_jump_position_final.py`) 现在都支持相同的压力测试参数。

- `--load_mass <float>`: 增加机身负载 (kg)。
- `--terrain`: 启用崎岖地形模式。
- `--disable_leg <leg_name>`: 禁用某条腿 (可选值: `FL`, `FR`, `RL`, `RR`)，模拟单腿失效。

**快速启动命令 (Quick Start Guide)**:
```bash
# 1. 实时预览 (Isaac Gym) - 加载 3月5日 3500代模型
python legged_gym/scripts/play.py --task go2_jump --load_run Mar05_07-28-45_ --checkpoint 3500

# 2. 鲁棒性压力测试 (Mujoco Sim2Sim - 力矩版本)
# 说明：加载5kg负载，在崎岖地形上测试，并禁用右前腿(FR)
# 注意：加载 exported 下的 policy_1.pt (由上述 play 命令导出)
python deploy_mujoco/sim2sim_GO2_jump_torque_final.py --load_model logs/go2_jump_torque/exported/policies/policy_1.pt --load_mass 5.0 --terrain --disable_leg FR
```

**基准对比测试 (Baseline Comparison)**:
为了验证力矩控制相对于位置控制的优势，可运行以下基于位置控制的模型进行对标，使用完全相同的压力测试参数。
```bash
# 3. 鲁棒性压力测试 (Mujoco Sim2Sim - 位置版 Baseline)
# 说明：同样加载5kg负载，在崎岖地形上测试，并禁用右前腿(FR)
python deploy_mujoco/sim2sim_GO2_jump_position_final.py --load_model logs/go2_jump/exported/policies/policy_1.pt --load_mass 5.0 --terrain --disable_leg FR
```

### 5.3 核心技术突破 (Key Technical Milestones)

1.  **动态相位释放 (Dynamic Phase Release)**: 引入随 PD 衰减进度自动释放的相位奖励，消除了传统硬性节拍导致的“空中弹腿”现象，使动作更符合自然动力学。
2.  **Sim2Sim 精度修复**: 解决了 Mujoco 环境中由于角速度坐标系定义不一致导致的转向（Yaw轴）失稳问题，实现了 1.0s 周期下的全向平稳跳跃。
3.  **姿态鲁棒性**: 通过动态增强对小腿（Calf）关节的约束，解决了 PD 退出后常见的“小腿收缩作弊”问题，确保了跳跃姿态的真实性。