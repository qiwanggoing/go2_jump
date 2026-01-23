from legged_gym.envs.Go2_MoB.GO2_JUMP.GO2_JUMP_config import GO2_JUMP_Cfg_Yu
import math
import numpy as np
import mujoco
import mujoco.viewer
from collections import deque
from scipy.spatial.transform import Rotation as R
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.utils import Logger
import torch
from pynput import keyboard
import yaml

# === 全局控制变量 ===
x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 0.0, 0.0, 0.0
x_vel_max, y_vel_max, yaw_vel_max = 1.5, 1.0, 3.0
speed_multiplier = 1.0

# Key states for "Hold to Move"
key_states = {
    'w': False, 's': False, 
    'a': False, 'd': False, 
    'q': False, 'e': False
}

# Surge/Jump State
surge_mode = False
surge_start_time = 0.0

def on_press(key):
    global speed_multiplier, surge_mode, surge_start_time
    try:
        if hasattr(key, 'char'):
            char = key.char.lower()
            if char in key_states:
                key_states[char] = True
            elif char == 'i': # 加速
                speed_multiplier = min(speed_multiplier * 1.1, 5.0)
            elif char == 'k': # 减速
                speed_multiplier = max(speed_multiplier * 0.9, 0.1)
            elif char == 'j': # Jump / Surge
                surge_mode = True
                surge_start_time = 0 # Will be set in main loop relative to sim time
                print("\n>>> SURGE COMMAND TRIGGERED! <<<")
    except AttributeError:
        pass

def on_release(key):
    try:
        if hasattr(key, 'char'):
            char = key.char.lower()
            if char in key_states:
                key_states[char] = False
    except AttributeError:
        pass

def quaternion_to_euler_array(quat):
    # Ensure quaternion is in the correct format [x, y, z, w]
    x, y, z, w = quat
    
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
    return np.array([roll_x, pitch_y, yaw_z])

def get_obs(data, model):
    '''Extracts an observation from the mujoco data structure'''
    q = data.qpos[7:19].astype(np.double)
    dq = data.qvel[6:].astype(np.double)
    # MuJoCo data.qpos[3:7] is [w, x, y, z]. Scipy R.from_quat expects [x, y, z, w].
    quat_scipy = data.qpos[3:7].astype(np.double)[[1, 2, 3, 0]] 
    
    r = R.from_quat(quat_scipy)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # In the base frame
    omega = data.qvel[3:6].astype(np.double) # In body frame
    gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
    base_pos = data.qpos[0:3].astype(np.double)
    
    foot_positions = []
    foot_forces = []
    # 这里不需要精确的 foot_positions 用于观测，只要跑通即可
    return (q, dq, quat_scipy, v, omega, gvec, base_pos, foot_positions, foot_forces)

def torque_control(residual_torque, q, kp, target_dq, dq, kd, cfg):
    '''
    Calculates torques using Residual Torque Control strategy.
    '''
    # 1. 基础 PD 力矩 (维持姿态, target_q 固定为 default)
    # 注意：这里的 kp 应该是很小的 (15.0)，主要靠 residual_torque
    pd_torque = (cfg.robot_config.default_dof_pos - q) * kp + (target_dq - dq) * kd
    
    # 2. 叠加残差力矩
    total_torque = pd_torque + residual_torque
    
    return total_torque

def run_mujoco(policy, cfg):
    global x_vel_cmd, y_vel_cmd, yaw_vel_cmd
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    
    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    model.opt.timestep = cfg.sim_config.dt
    data = mujoco.MjData(model)
    
    num_actuated_joints = cfg.env.num_actions
    data.qpos[-num_actuated_joints:] = cfg.robot_config.default_dof_pos
    mujoco.mj_step(model, data)
    
    count_lowlevel = 0
    
    # 预热 buffer
    hist_obs = deque()
    for _ in range(cfg.env.frame_stack):
        hist_obs.append(np.zeros([1, cfg.env.num_single_obs], dtype=np.double))
    
    action = np.zeros((cfg.env.num_actions), dtype=np.double)
    target_dq = np.zeros((cfg.env.num_actions), dtype=np.double) # 目标速度始终为0

    # Metrics
    max_height = 0.0
    max_speed = 0.0
    
    # Handle Surge timing
    global surge_mode, surge_start_time

    with mujoco.viewer.launch_passive(model, data) as viewer:
        import time
        start_time = time.time()
        
        while viewer.is_running():
            step_start = time.time()
            current_sim_time = time.time() - start_time

            # 1. 获取观测
            q, dq, quat, v, omega, gvec, base_pos, _, _ = get_obs(data, model)

            # === Dashboard & Metrics ===
            current_height = base_pos[2]
            current_speed = np.linalg.norm(v[:2])
            if current_height > max_height: max_height = current_height
            if current_speed > max_speed: max_speed = current_speed

            if count_lowlevel % 20 == 0:
                print(f"\r[Perf] H: {current_height:.3f}m (Max: {max_height:.3f}) | V: {current_speed:.2f}m/s (Max: {max_speed:.2f}) | Mult: {speed_multiplier:.1f}x ", end="")

            # === Command Logic (WASD + Surge) ===
            # Normal WASD
            target_vx = (1.0 if key_states['w'] else 0.0) - (1.0 if key_states['s'] else 0.0)
            target_vy = (1.0 if key_states['a'] else 0.0) - (1.0 if key_states['d'] else 0.0)
            target_vyaw = (1.0 if key_states['q'] else 0.0) - (1.0 if key_states['e'] else 0.0)
            
            target_vx *= x_vel_max * speed_multiplier
            target_vy *= y_vel_max * speed_multiplier
            target_vyaw *= yaw_vel_max * speed_multiplier

            # Surge Override
            if surge_mode:
                if surge_start_time == 0: surge_start_time = current_sim_time
                
                if current_sim_time - surge_start_time < 0.4: # Surge duration 0.4s
                    target_vx = 2.5 # Violent forward push
                    target_vy = 0.0
                    target_vyaw = 0.0
                else:
                    surge_mode = False
                    surge_start_time = 0

            # Smooth update
            alpha = 0.1 if surge_mode else 0.05 
            x_vel_cmd = x_vel_cmd * (1 - alpha) + target_vx * alpha
            y_vel_cmd = y_vel_cmd * (1 - alpha) + target_vy * alpha
            yaw_vel_cmd = yaw_vel_cmd * (1 - alpha) + target_vyaw * alpha
            # ==================================================
            
            # 2. 策略推理 (控制频率 decimation)
            
            # 2. 策略推理 (控制频率 decimation)
            if count_lowlevel % cfg.sim_config.decimation == 0:
                obs = np.zeros([1, cfg.env.num_single_obs], dtype=np.float32)
                eu_ang = quaternion_to_euler_array(quat)
                eu_ang[eu_ang > math.pi] -= 2 * math.pi

                # 填充 Observation
                obs[0, 0] = math.sin(2 * math.pi * count_lowlevel * cfg.sim_config.dt / cfg.rewards.cycle_time)
                obs[0, 1] = math.cos(2 * math.pi * count_lowlevel * cfg.sim_config.dt / cfg.rewards.cycle_time)
                obs[0, 2] = x_vel_cmd * cfg.normalization.obs_scales.lin_vel
                obs[0, 3] = y_vel_cmd * cfg.normalization.obs_scales.lin_vel
                obs[0, 4] = yaw_vel_cmd * cfg.normalization.obs_scales.ang_vel
                
                obs[0, 5:8] = omega * cfg.normalization.obs_scales.ang_vel
                obs[0, 8:11] = eu_ang * cfg.normalization.obs_scales.quat
                
                obs[0, 11:23] = (q - cfg.robot_config.default_dof_pos) * cfg.normalization.obs_scales.dof_pos
                obs[0, 23:35] = dq * cfg.normalization.obs_scales.dof_vel
                
                obs[0, 35:47] = action # Last Action

                obs = np.clip(obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)

                hist_obs.append(obs)
                hist_obs.popleft()

                policy_input = np.zeros([1, cfg.env.num_observations], dtype=np.float32)
                for i in range(cfg.env.frame_stack):
                    policy_input[0, i * cfg.env.num_single_obs : (i + 1) * cfg.env.num_single_obs] = hist_obs[i][0, :]

                # 推理
                action[:] = policy(torch.tensor(policy_input))[0].detach().numpy()
                action = np.clip(action, -cfg.normalization.clip_actions, cfg.normalization.clip_actions)
            
            # 3. 计算力矩 (Low Level Control)
            # [关键] Action 直接乘以 Scale 变成 Residual Torque
            residual_torque = action * cfg.control.action_scale
            
            tau = torque_control(residual_torque, q, cfg.robot_config.kps, 
                                 target_dq, dq, cfg.robot_config.kds, cfg)
            
            tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)
            
            data.ctrl = tau
            mujoco.mj_step(model, data)
            
            count_lowlevel += 1
            viewer.sync()

            # 保持实时性
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Deployment script for Torque Control.')
    # 默认加载最新的 policy，你需要确认路径对不对
    parser.add_argument('--load_model', type=str, required=True, help='Path to the torque policy model .pt file')
    parser.add_argument('--terrain', action='store_true', help='terrain or plane')
    args = parser.parse_args()

    policy = torch.jit.load(args.load_model)
    run_mujoco(policy, Sim2simCfg())