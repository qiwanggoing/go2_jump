import sys
from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger
import numpy as np
import torch

def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # 强制单机模式
    env_cfg.env.num_envs = 1
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.commands.resampling_time = 1e9 

    env_cfg.env.test = True
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    
    # 同步 PD 衰减进度
    if args.checkpoint != -1:
        try:
            iter_num = int(args.checkpoint)
            steps_per_iter = train_cfg.runner.num_steps_per_env
            env.common_step_counter = iter_num * steps_per_iter
            print(f"[MaxVelTest] Syncing curriculum: Set common_step_counter to {env.common_step_counter}")
        except:
            print("[MaxVelTest] Warning: Could not parse checkpoint.")
            
    obs = env.get_observations()
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # 测试参数
    current_cmd_vel = 0.5
    vel_increment = 0.1
    warmup_steps = 400    # 延长加速适应期 (8s)
    record_steps = 400    # 延长数据采集期 (8s)
    cycle_steps = warmup_steps + record_steps
    
    prev_avg_vel = 0.0
    vel_history = []
    contact_history = []
    max_test_vel = 4.0
    
    cycle_idx = 0
    total_steps = 0

    print(f"\n{'='*50}\nStarting Automated Max Velocity Test (Enhanced)\n{'='*50}")

    while True:
        # 1. 指令更新与周期统计
        if cycle_idx >= cycle_steps:
            # 一个周期结束，计算上一周期的平均速度
            avg_vel = np.mean(vel_history) if len(vel_history) > 0 else 0.0
            vel_inc = avg_vel - prev_avg_vel
            
            # 计算跳跃频率 (通过触地状态切换)
            # contact_history 是 [steps, 4] 的布尔矩阵
            if len(contact_history) > 1:
                contacts = np.array(contact_history)
                # 统计四足中任意一只脚从 Air -> Stance 的切换
                switches = np.sum((contacts[1:] > 0.5) & (contacts[:-1] < 0.5))
                # 归一化到单足频率 (四足跳跃通常是同步或交替)
                jump_freq = (switches / 4.0) / (len(contact_history) * env.dt)
            else:
                jump_freq = 0.0

            print(f"\n>>> Cycle Results: Cmd {current_cmd_vel:.2f} | Avg Actual: {avg_vel:.3f} | Jump Freq: {jump_freq:.2f} Hz | Inc: {vel_inc:.3f}")
            
            # 饱和判定：如果真实速度增加不明显
            if vel_inc < 0.02 and total_steps > cycle_steps:
                print(f"\n[SATURATION] Velocity reached physical limit at Cmd: {current_cmd_vel:.2f} m/s")
                break
            
            # 准备下一档速度
            prev_avg_vel = avg_vel
            vel_history = []
            contact_history = []
            current_cmd_vel += vel_increment
            cycle_idx = 0
            if current_cmd_vel > max_test_vel: break
            print(f"Increasing Command to: {current_cmd_vel:.2f} m/s")

        env.commands[:, 0] = current_cmd_vel
        env.commands[:, 1] = 0.
        env.commands[:, 2] = 0.

        # 2. 运行推理与环境步进
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())
        
        # 3. 处理重置逻辑 (修复 1200 步退出问题)
        if dones[0]:
            if "time_outs" in infos and infos["time_outs"][0]:
                print(f"\n[TIMEOUT] Environment reset at step {total_steps}. Restarting current cycle for Cmd {current_cmd_vel:.2f}...")
                cycle_idx = 0 # 重置当前速度档位的测试周期
                vel_history = []
                contact_history = []
                continue # 继续循环，不累加 cycle_idx
            else:
                print(f"\n[FALL] Robot fell at Command Velocity: {current_cmd_vel:.2f} m/s")
                print(f"Last stable average velocity was: {prev_avg_vel:.3f} m/s")
                break

        # 4. 数据采集 (仅在采集期记录)
        actual_vel = env.base_lin_vel[0, 0].item()
        # 检查脚部触地力 (Index 2 是 Z 轴力)
        foot_contacts = (env.contact_forces[0, env.feet_indices, 2] > 1.0).cpu().numpy()
        
        if cycle_idx >= warmup_steps:
            vel_history.append(actual_vel)
            contact_history.append(foot_contacts)

        # 5. 视角跟随 (SATA 风格)
        robot_pos = env.root_states[0, :3].cpu().numpy()
        camera_position = robot_pos + np.array([-2.5, 2.0, 1.2]) # 稍微拉远一点方便观察整体姿态
        env.set_camera(camera_position, robot_pos)

        # 6. 实时打印
        if total_steps % 50 == 0:
            phase = "WARMUP" if cycle_idx < warmup_steps else "RECORD"
            print(f"Step {total_steps:5d} | [{phase}] | Cmd: {current_cmd_vel:.2f} | Actual: {actual_vel:.3f} | Contacts: {np.sum(foot_contacts)}")

        cycle_idx += 1
        total_steps += 1

if __name__ == '__main__':
    args = get_args()
    play(args)
