
# Continuous Versatile Jumping Using Learned Action Residuals  
## 可复现级技术笔记（Implementation-Oriented Rewrite）

> 本文档目标：**读者可根据本文档独立复现论文中的方法**，包括系统结构、控制逻辑、强化学习设置与关键工程细节。  
> 论文来源：Yang et al., 2023（L4DC）

---

## 0. 任务定义（Task Definition）

**目标**：  
控制四足机器人（Unitree Go1）实现：

- 连续跳跃（连续多个 jump episode）
- 多方向跳跃（前 / 后 / 左 / 右 / 任意角度插值）
- 跳跃过程中可叠加 **yaw 转向**
- 可直接从仿真迁移到真实机器人（zero-shot sim2real）

**基本假设**：

- 跳跃采用 **pronking gait**
- 所有足同时离地 / 同时着地
- 跳跃周期固定为 **1.0 s**
  - stance：0.5 s
  - swing（flight）：0.5 s

---

## 1. 系统总体架构（Full Control Stack）

控制系统采用 **三层结构**：

```
[ High-level RL Residual Policy ]
            ↓
[ Stance / Swing Controllers ]
            ↓
[ Whole-Body Controller (WBC) ]
            ↓
[ Motor Torques ]
```

并行模块：

- Contact Scheduler（时间驱动）
- State Estimator（Kalman Filter）

**控制频率**：500 Hz（仿真 & 实机一致）

---

## 2. Contact Scheduler（接触调度器）

### 2.1 设计

- 开环、基于时间
- 不依赖 RL
- 所有腿共享同一相位

```text
phase ∈ [0, 1)

if phase < 0.5:
    stance phase (all legs contact)
else:
    swing phase (all legs airborne)
```

### 2.2 作用

- 明确奖励函数中的“期望接触状态”
- 避免 RL 学习接触时序（极大降低搜索难度）

---

## 3. 状态估计（State Estimation）

使用 Kalman Filter，输出：

- base position (x, y, z)
- base orientation (roll, pitch, yaw)
- base linear velocity
- base angular velocity
- foot positions（forward kinematics）

⚠️ **关键点**：  
WBC 对状态噪声极其敏感，必须保证估计稳定。

---

## 4. 支撑相控制器（Stance Controller）

### 4.1 核心思想

> **RL 不直接控制机器人，而是“修正”一个可工作的解析控制器**

最终输出：

```
a_body = a_acc_controller + a_residual_policy
```

---

## 5. 手工加速度控制器（Acceleration Controller）

### 5.1 简化动力学模型

- 将机器人视为 **质点**
- 忽略角动量耦合
- 仅考虑 CoM 平移 + yaw

### 5.2 起跳速度计算（Lift-off Velocity）

给定：

- 期望落点位移：(p_x, p_y)
- 期望 yaw 变化：p_yaw
- swing 时间：t_swing = 0.5 s

计算：

```
v_x   = p_x   / t_swing
v_y   = p_y   / t_swing
v_yaw = p_yaw / t_swing
v_z   = 0.5 * g * t_swing
```

其中：

- g = 9.81 m/s²

### 5.3 加速度跟踪律

在 stance 剩余时间 t 内：

```
a_des = (v_liftoff - v_current) / t
```

### 5.4 运动学可行性检查（关键工程细节）

通过数值积分预测 lift-off 时 CoM 位置：

```
x_liftoff = x_now + v_now * t + 0.5 * a_des * t²
```

若超出经验运动学边界（bounding box）：

- 不执行跳跃
- 转而移动至 **low-standing preparation pose**

⚠️ **这是防止 RL 产生“不可落地跳跃”的关键模块**

---

## 6. 残差策略（Residual RL Policy）

### 6.1 作用

- 修正 pitch / roll 漂移
- 补偿简化模型误差
- 稳定空中姿态

### 6.2 状态空间（Observation）

```
s = [
  base_pos (3)
  base_ori (roll, pitch, yaw) (3)
  base_lin_vel (3)
  base_ang_vel (3)
  foot_pos (12)
  relative_target_pos (3)
  remaining_phase_time (1)
]
```

维度 ≈ 28 ~ 30

---

## 7. 动作空间（Action Space）

⚠️ **这是可复现的关键设计点**

策略输出：

```
a = [
  linear_acc_x
  linear_acc_y
  linear_acc_z
  angular_acc_z
  desired_roll
  desired_pitch
]
```

未由策略控制的量：

- yaw position
- angular acc x/y
- linear velocity references

均由启发式或当前状态直接给定

---

## 8. 奖励函数（Reward Function）

```
r = r_alive
  + w_p * r_position
  + w_o * r_orientation
  + w_c * r_contact
```

### 8.1 各项定义

- **Alive bonus**
  ```
  r_alive = +4
  ```

- **落点误差**
  ```
  r_position = - ||p_current - p_target||² / ||p_target||²
  ```

- **姿态惩罚**
  ```
  r_orientation = -(roll² + pitch²)
  ```

- **接触一致性**
  ```
  r_contact = Σ I(c_i != ĉ_i)
  ```

权重：

```
w_p = 1.0
w_o = 5.0
w_c = 0.4
```

---

## 9. Early Termination 条件

episode 立即终止若：

- base height < 0.08 m
- cos(upright, gravity) < 0.6
- 非足端部位接触地面

👉 **极大提高训练稳定性与效率**

---

## 10. 强化学习算法（ARS）

### 10.1 为什么不用 PPO / SAC？

- 接触切换 → 非平滑 reward
- 分层控制 → 非 Markov
- WBC 对高频噪声极度敏感

### 10.2 ARS 设置

- policy: 1 hidden layer, 256 units, tanh
- 参数空间高斯扰动
- 并行 rollout
- 训练时间：~3 小时（16 核 CPU）

---

## 11. Whole-Body Controller（WBC）

### 11.1 输入

- p_body, v_body, a_body（6 DoF）
- p_foot（摆动足）

### 11.2 输出

- motor position q̄
- motor velocity q̄_dot
- feedforward torque τ̄

### 11.3 执行

```
τ = kp (q̄ - q) + kd (q̄_dot - q_dot) + τ̄
```

---

## 12. 仿真与真实部署关键一致性

| 项目 | 仿真 | 实机 |
|----|----|----|
| 控制频率 | 500 Hz | 500 Hz |
| WBC | 相同 | 相同 |
| 接触调度 | 相同 | 相同 |
| Policy | 相同 | 相同 |
| 微调 | ❌ | ❌ |

---

## 13. 复现 Checklist（强烈建议）

- [ ] 固定 jump timing
- [ ] 先实现 acceleration controller
- [ ] 单独测试 WBC 跟踪精度
- [ ] 再加入 residual RL
- [ ] 不要让 RL 控制接触
- [ ] 必须有 early termination

---

## 14. 总结

**一句话总结**：  
> 这是一个“控制器先保证可跳，RL 再让它跳得好”的工程范式。

该思路非常适合：

- 高动态技能
- 强物理约束任务
- 需要 sim2real 的机器人系统
