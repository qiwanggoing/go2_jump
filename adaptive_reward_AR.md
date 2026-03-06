
# 自适应奖励（Adaptive Reward, AR）设计说明

本文档总结并整理了 **Xue et al., 2025 – Learning robust quadrupedal locomotion under disturbances** 中提出的 **自适应奖励（Adaptive Reward, AR）** 方法，并结合当前 **PD + torque bias → 纯力矩控制** 的训练目标，给出可直接落地的实现形式，适合作为论文方法章节或实现说明。

---

## 1. 设计动机

在存在外界扰动、接触不确定性和高维力矩动作空间的情况下，**传统线性加和奖励**存在两个主要问题：

1. 负奖励（碰撞、姿态偏差、能耗等）在训练早期可能压制探索；
2. 在后期训练中，奖励权重难以同时兼顾“任务性能”和“动作稳定性”。

Xue et al.（2025）提出的 **Adaptive Reward（AR）** 通过**指数门控（exponential attenuation）**的方式，引入一个**随训练进度自适应变化的约束强度**，从而实现：

- 训练早期：鼓励探索、容忍较大误差；
- 训练后期：强化稳定性与安全性约束。

---

## 2. 奖励结构分解

在 AR 设计中，总奖励被拆分为两部分：

### 2.1 正奖励项（r_pos）

正奖励项表示**任务相关的性能指标**，通常为非负量，例如：

- 速度/跳跃高度/前进距离跟踪
- 成功完成跳跃或站稳
- 期望运动方向一致性

形式上：

\[
r_{pos} = \sum_i w_i^+ \, r_i^+
\]

其中 \( r_i^+ \ge 0 \)。

---

### 2.2 负奖励项（r_neg）

负奖励项表示**需要被约束或抑制的行为**，例如：

- 机身姿态偏差（roll / pitch）
- 碰撞或非法接触
- 力矩幅值、力矩变化率（torque rate）
- 动作抖动、关节越界

在 AR 中，这些负项**不直接线性相加**，而是构成一个能量项：

\[
E_{neg} = \sum_j (w_j^- \, r_j^-)^2
\]

其中 \( r_j^- \ge 0 \)。

---

## 3. 自适应奖励（Adaptive Reward）公式

### 3.1 基础指数门控奖励

Xue et al. 首先提出如下形式的奖励整合方式：

\[
r = r_{pos} \cdot \exp\left( - \frac{E_{neg}}{\sigma} \right)
\]

其中：

- \( E_{neg} \)：负奖励能量
- \( \sigma \)：控制惩罚强度的尺度参数

该形式的特点是：

- 当负项较小时，对正奖励影响较小；
- 当负项增大时，总奖励会被**指数级抑制**，而非线性下降。

---

### 3.2 引入自适应因子 ξ

为了让约束强度**随训练进度自动变化**，论文进一步引入一个 **能力/进度指标** \( \xi \in [0,1] \)，并将奖励改写为：

\[
r = r_{pos} \cdot \exp\left(
- \frac{E_{neg}}{(1 - \xi)\, \sigma}
\right)
\]

其中：

- 训练早期：\( \xi \approx 0 \)，惩罚较弱，鼓励探索；
- 训练后期：\( \xi \rightarrow 1 \)，惩罚显著增强，强化稳定性。

---

## 4. ξ（能力指标）的实现方式

论文中使用 **Adaptive Evaluation Mechanism (AEM)** 来估计 \( \xi \)，该机制基于多个 episode 级指标。

在工程实现中，可采用更简化但有效的定义，例如：

- 成功 episode 比例
- 未跌倒时间占比
- 姿态误差的归一化反比
- 跳跃高度或速度跟踪误差的滑动平均

示例：

\[
\xi = \text{clip}\left(
1 - \frac{\bar{e}_{task}}{e_{max}}, \, 0, \, 1
\right)
\]

其中 \( \bar{e}_{task} \) 为 episode 级平均任务误差。

---

## 5. 数值稳定的实现形式（推荐）

为避免数值问题，实际实现中推荐：

- 对分母加下界 \( \varepsilon \)
- 对负项进行加权平方

```python
neg_energy = sum((w_j * r_neg_j)**2 for j in neg_terms)
denom = max(eps, (1 - xi) * sigma)
attenuation = exp(-neg_energy / denom)
r_total = r_pos * attenuation
```

---

## 6. 与纯力矩控制训练的关系

在从 **PD + torque bias** 过渡到 **纯 torque policy** 的过程中：

- 力矩空间更大，早期探索更不稳定；
- 传统线性奖励容易导致训练崩溃。

AR 的优势在于：

- 在早期允许较大力矩波动与姿态误差；
- 在策略能力提升后，自动强化对
  - 碰撞
  - 姿态
  - 力矩幅值与变化率
  的约束。

因此，AR 特别适合作为 **纯力矩控制策略训练的核心奖励结构**。

---

## 7. 推荐使用方式总结

1. 将奖励拆分为 `r_pos` 与 `r_neg`；
2. 使用指数门控而非线性相加；
3. 引入 episode 级能力指标 \( \xi \)；
4. 与 PD/力矩先验的**逐步衰减策略**结合，可实现稳定过渡到纯力矩控制。

---

**参考文献**  
Xue et al., *Learning robust quadrupedal locomotion under disturbances via reinforcement learning with an autonomous evaluation mechanism*, 2025.
