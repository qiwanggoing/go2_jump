from legged_gym.envs.Go2_MoB.GO2_JUMP.GO2_JUMP_config import GO2_JUMP_Cfg_Yu
import math
import numpy as np
import mujoco
import mujoco.viewer
from collections import deque
from scipy.spatial.transform import Rotation as R
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
from pynput import keyboard

# === 全局控制变量 ===
x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 0.0, 0.0, 0.0
x_vel_max, y_vel_max, yaw_vel_max = 1.5, 1.0, 3.0
speed_multiplier = 1.0

key_states = {'w': False, 's': False, 'a': False, 'd': False, 'q': False, 'e': False}
surge_mode = False
surge_start_time = 0.0

def on_press(key):
    global speed_multiplier, surge_mode, surge_start_time
    try:
        if hasattr(key, 'char'):
            char = key.char.lower()
            if char in key_states: key_states[char] = True
            elif char == 'i': speed_multiplier = min(speed_multiplier * 1.1, 5.0)
            elif char == 'k': speed_multiplier = max(speed_multiplier * 0.9, 0.1)
            elif char == 'j':
                surge_mode = True
                surge_start_time = 0
                print("\n>>> SURGE COMMAND TRIGGERED! <<<")
    except AttributeError: pass

def on_release(key):
    try:
        if hasattr(key, 'char'):
            char = key.char.lower()
            if char in key_states: key_states[char] = False
    except AttributeError: pass

def get_obs(data, model):
    '''提取符合 Isaac Gym 逻辑的观测'''
    q = data.qpos[7:19].astype(np.double)
    dq = data.qvel[6:].astype(np.double)
    
    # 四元数转换 [w, x, y, z] -> [x, y, z, w]
    quat_scipy = data.qpos[3:7].astype(np.double)[[1, 2, 3, 0]] 
    r = R.from_quat(quat_scipy)
    
    # 1. 旋转角速度到本体坐标系 (Isaac Gym 核心对齐)
    omega = r.apply(data.qvel[3:6], inverse=True).astype(np.double) 
    
    # 2. 转换欧拉角 [Roll, Pitch, Yaw] (修复 as_euler 报错)
    eu_ang = r.as_euler('xyz', degrees=False).astype(np.double)
    eu_ang = (eu_ang + np.pi) % (2 * np.pi) - np.pi
    
    base_pos = data.qpos[0:3].astype(np.double)
    return q, dq, quat_scipy, omega, eu_ang, base_pos

def torque_control(residual_torque, q, target_q, kp, dq, kd):
    '''Isaac Gym PD 计算结构'''
    return kp * (target_q - q) - kd * dq + residual_torque

def run_mujoco(policy, cfg):
    global x_vel_cmd, y_vel_cmd, yaw_vel_cmd, surge_mode, surge_start_time
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    
    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    model.opt.timestep = cfg.sim_config.dt
    data = mujoco.MjData(model)
    
    # 初始姿态对齐
    data.qpos[7:19] = cfg.robot_config.default_dof_pos
    mujoco.mj_step(model, data)
    
    count_lowlevel = 0
    hist_obs = deque(maxlen=cfg.env.frame_stack)
    for _ in range(cfg.env.frame_stack):
        hist_obs.append(np.zeros([1, cfg.env.num_single_obs], dtype=np.double))
    
    action = np.zeros(cfg.env.num_actions, dtype=np.double)
    tau = np.zeros(cfg.env.num_actions, dtype=np.double)

    # === [关键对齐] Iter 4000 部署参数 ===
    progress = 1.0 
    current_action_scale = 23.5 
    pd_factor = 0.0 # 纯力矩模式

    with mujoco.viewer.launch_passive(model, data) as viewer:
        import time
        start_time = time.time()
        
        while viewer.is_running():
            step_start = time.time()
            current_sim_time = time.time() - start_time

            # 1. 获取观测
            q, dq, quat, omega, eu_ang, base_pos = get_obs(data, model)

            if count_lowlevel % 20 == 0:
                print(f"\r[Perf] H: {base_pos[2]:.3f}m | V: {np.linalg.norm(data.qvel[:2]):.2f}m/s | Progress: {progress:.2f} ", end="")

            # 指令逻辑
            target_vx = (1.0 if key_states['w'] else 0.0) - (1.0 if key_states['s'] else 0.0)
            target_vy = (1.0 if key_states['a'] else 0.0) - (1.0 if key_states['d'] else 0.0)
            target_vyaw = (1.0 if key_states['q'] else 0.0) - (1.0 if key_states['e'] else 0.0)
            
            if surge_mode:
                if surge_start_time == 0: surge_start_time = current_sim_time
                if current_sim_time - surge_start_time < 0.4: target_vx = 1.0
                else: surge_mode, surge_start_time = False, 0

            # 保持基本的平滑，对齐 Isaac Gym 表现
            alpha = 0.05
            x_vel_cmd = x_vel_cmd * (1 - alpha) + target_vx * x_vel_max * speed_multiplier * alpha
            y_vel_cmd = y_vel_cmd * (1 - alpha) + target_vy * y_vel_max * speed_multiplier * alpha
            yaw_vel_cmd = yaw_vel_cmd * (1 - alpha) + target_vyaw * yaw_vel_max * speed_multiplier * alpha
            
            # 2. 策略推理 (50Hz)
            if count_lowlevel % cfg.sim_config.decimation == 0:
                obs = np.zeros([1, cfg.env.num_single_obs], dtype=np.float32)

                # 填充 60 维 Observation
                obs[0, 0] = math.sin(2 * math.pi * count_lowlevel * cfg.sim_config.dt / cfg.rewards.cycle_time)
                obs[0, 1] = math.cos(2 * math.pi * count_lowlevel * cfg.sim_config.dt / cfg.rewards.cycle_time)
                obs[0, 2] = x_vel_cmd * cfg.normalization.obs_scales.lin_vel
                obs[0, 3] = y_vel_cmd * cfg.normalization.obs_scales.lin_vel
                obs[0, 4] = yaw_vel_cmd * cfg.normalization.obs_scales.ang_vel
                obs[0, 5:8] = omega * cfg.normalization.obs_scales.ang_vel
                obs[0, 8:11] = eu_ang * cfg.normalization.obs_scales.quat
                obs[0, 11:23] = (q - cfg.robot_config.default_dof_pos) * cfg.normalization.obs_scales.dof_pos
                obs[0, 23:35] = dq * cfg.normalization.obs_scales.dof_vel
                obs[0, 35:47] = action 
                obs[0, 47:59] = tau    
                obs[0, 59] = progress

                obs = np.clip(obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)
                hist_obs.append(obs)
                
                policy_input = np.zeros([1, cfg.env.num_observations], dtype=np.float32)
                for i in range(cfg.env.frame_stack):
                    policy_input[0, i * cfg.env.num_single_obs : (i + 1) * cfg.env.num_single_obs] = hist_obs[i][0, :]

                # 获取 Action
                action[:] = policy(torch.tensor(policy_input))[0].detach().numpy()
                action = np.clip(action, -cfg.normalization.clip_actions, cfg.normalization.clip_actions)
            
            # 3. 计算力矩 (200Hz)
            residual_torque = action * current_action_scale
            tau = torque_control(residual_torque, q, cfg.robot_config.default_dof_pos, 
                                 cfg.robot_config.kps * pd_factor, dq, cfg.robot_config.kds * pd_factor)
            
            tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)
            data.ctrl = tau
            mujoco.mj_step(model, data)
            count_lowlevel += 1
            viewer.sync()

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0: time.sleep(time_until_next_step)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Deployment script for Torque Control.')
    parser.add_argument('--load_model', type=str, required=True, help='Path to the torque policy model .pt file')
    parser.add_argument('--terrain', action='store_true', help='terrain or plane')
    args = parser.parse_args()

    class Sim2simCfg(GO2_JUMP_Cfg_Yu):
        class sim_config:
            mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/go2/scene{"_terrain" if args.terrain else ""}.xml'
            dt, decimation = 0.005, 4
        class robot_config:
            kps, kds = np.full(12, 20.0), np.full(12, 0.5)
            tau_limit = np.array([23.7, 23.7, 35.55] * 4)
            default_dof_pos = np.array([0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5])

    policy = torch.jit.load(args.load_model)
    run_mujoco(policy, Sim2simCfg())
