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
current_speed_target = 1.0 # 全局速度档位

key_states = {'w': False, 's': False, 'a': False, 'd': False, 'q': False, 'e': False}
surge_mode = False
surge_start_time = 0.0

def on_press(key):
    global current_speed_target, surge_mode, surge_start_time
    try:
        if hasattr(key, 'char'):
            char = key.char.lower()
            if char in key_states: key_states[char] = True
            elif char == 'i': 
                current_speed_target = min(current_speed_target + 0.1, 5.0)
                print(f"\n>>> Speed Target Increased: {current_speed_target:.1f} m/s <<<")
            elif char == 'k': 
                current_speed_target = max(current_speed_target - 0.1, 0.0)
                print(f"\n>>> Speed Target Decreased: {current_speed_target:.1f} m/s <<<")
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

def quaternion_to_euler_array(quat):
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
    '''提取符合 Isaac Gym 逻辑的观测'''
    q = data.qpos[7:19].astype(np.double)
    dq = data.qvel[6:].astype(np.double)
    q_raw = data.qpos[3:7].astype(np.double) 
    quat_scipy = q_raw[[1, 2, 3, 0]] 
    omega = data.qvel[3:6].astype(np.double) 
    eu_ang = quaternion_to_euler_array(quat_scipy)
    eu_ang[eu_ang > math.pi] -= 2 * math.pi
    base_pos = data.qpos[0:3].astype(np.double)
    return q, dq, quat_scipy, omega, eu_ang, base_pos

def torque_control(residual_torque, q, target_q, kp, dq, kd):
    return kp * (target_q - q) - kd * dq + residual_torque

def run_mujoco(policy, cfg):
    global x_vel_cmd, y_vel_cmd, yaw_vel_cmd, surge_mode, surge_start_time
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    
    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    model.opt.timestep = cfg.sim_config.dt
    data = mujoco.MjData(model)
    data.qpos[7:19] = cfg.robot_config.default_dof_pos
    mujoco.mj_step(model, data)
    
    count_lowlevel = 0
    hist_obs = deque(maxlen=cfg.env.frame_stack)
    for _ in range(cfg.env.frame_stack):
        hist_obs.append(np.zeros([1, cfg.env.num_single_obs], dtype=np.double))
    
    action = np.zeros(cfg.env.num_actions, dtype=np.double)
    last_tau = np.zeros(cfg.env.num_actions, dtype=np.double) 

    # === 性能监测变量 ===
    max_vel_reached = 0.0
    total_abs_error_x = 0.0
    tracking_steps = 0
    progress = 1.0 
    current_action_scale = 23.5 
    pd_factor = 0.0 

    with mujoco.viewer.launch_passive(model, data) as viewer:
        import time
        start_time = time.time()
        last_hz_check_time = start_time
        hz_counter = 0
        real_hz = 0.0

        while viewer.is_running():
            step_start = time.time()
            current_sim_time = time.time() - start_time
            hz_counter += 1
            if time.time() - last_hz_check_time > 1.0:
                real_hz = hz_counter / (time.time() - last_hz_check_time)
                hz_counter = 0
                last_hz_check_time = time.time()

            # 1. 获取观测
            q, dq, quat, omega, eu_ang, base_pos = get_obs(data, model)
            actual_vel_x = data.qvel[0] # 获取 X 轴实际速度
            actual_vel_norm = np.linalg.norm(data.qvel[:2])
            if actual_vel_norm > max_vel_reached: max_vel_reached = actual_vel_norm

            # === [核心对齐] MAE 统计逻辑 ===
            # 条件：运行超过 2s 且 X 指令大于 0.5 (排除站立和启动期)
            if current_sim_time > 2.0 and np.abs(x_vel_cmd) > 0.5:
                total_abs_error_x += np.abs(x_vel_cmd - actual_vel_x)
                tracking_steps += 1

            mae_x = total_abs_error_x / max(tracking_steps, 1)

            # 指令逻辑

            target_vx = (1.0 if key_states['w'] else 0.0) - (1.0 if key_states['s'] else 0.0)
            target_vy = (1.0 if key_states['a'] else 0.0) - (1.0 if key_states['d'] else 0.0)
            target_vyaw = (1.0 if key_states['q'] else 0.0) - (1.0 if key_states['e'] else 0.0)
            
            if surge_mode:
                curr_t = time.time() - start_time
                if surge_start_time == 0: surge_start_time = curr_t
                if curr_t - surge_start_time < 0.4: target_vx = 1.0
                else: surge_mode, surge_start_time = False, 0

            alpha = 0.05
            x_vel_cmd = x_vel_cmd * (1 - alpha) + target_vx * current_speed_target * alpha
            y_vel_cmd = y_vel_cmd * (1 - alpha) + target_vy * current_speed_target * alpha
            yaw_vel_cmd = yaw_vel_cmd * (1 - alpha) + target_vyaw * current_speed_target * alpha
            
            # 2. 策略推理 (200 Hz when decimation == 1)
            if count_lowlevel % cfg.sim_config.decimation == 0:
                obs = np.zeros([1, cfg.env.num_single_obs], dtype=np.float32)
                phase = 2 * math.pi * count_lowlevel * cfg.sim_config.dt / cfg.rewards.cycle_time
                obs[0, 0], obs[0, 1] = math.sin(phase), math.cos(phase)
                
                # 恢复使用配置中的 scale，并确保 index 对齐
                obs[0, 2] = x_vel_cmd * cfg.normalization.obs_scales.lin_vel
                obs[0, 3] = y_vel_cmd * cfg.normalization.obs_scales.lin_vel
                obs[0, 4] = yaw_vel_cmd * cfg.normalization.obs_scales.ang_vel
                obs[0, 5:8] = omega * cfg.normalization.obs_scales.ang_vel
                eu_ang_fixed = eu_ang.copy(); eu_ang_fixed[2] = 0.0
                obs[0, 8:11] = eu_ang_fixed * cfg.normalization.obs_scales.quat
                
                q_norm = (q - cfg.robot_config.default_dof_pos) * cfg.normalization.obs_scales.dof_pos
                dq_norm = dq * cfg.normalization.obs_scales.dof_vel
                obs[0, 11:35] = np.concatenate([q_norm, dq_norm])
                
                obs[0, 35:47] = action   
                obs[0, 47:59] = last_tau 
                obs[0, 59]    = progress 

                hist_obs.append(np.clip(obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations))
                policy_input = np.zeros([1, cfg.env.num_observations], dtype=np.float32)
                for i in range(cfg.env.frame_stack):
                    policy_input[0, i * cfg.env.num_single_obs : (i + 1) * cfg.env.num_single_obs] = hist_obs[i][0, :]

                action[:] = policy(torch.tensor(policy_input))[0].detach().numpy()
                action = np.clip(action, -cfg.normalization.clip_actions, cfg.normalization.clip_actions)
                
                # Keep rendering tied to the policy update path.
                viewer.sync()

            # 3. 计算并施加力矩 (200 Hz)
            # 还原为瞬时动作，移除延迟缓冲区
            tau = torque_control(action * current_action_scale, q, cfg.robot_config.default_dof_pos, 
                                 cfg.robot_config.kps * pd_factor, dq, cfg.robot_config.kds * pd_factor)
            
            tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)
            last_tau[:] = tau 
            
            if count_lowlevel % 20 == 0:
                print(f"\r[Perf] V: {actual_vel_norm:.2f} | MaxV: {max_vel_reached:.2f} | MAE: {mae_x:.3f} | Hz: {real_hz:.1f} | Torque: {np.max(np.abs(tau)):.1f}Nm ", end="")

            data.ctrl = tau
            mujoco.mj_step(model, data)
            count_lowlevel += 1

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
            dt, decimation = 0.005, 1
        class robot_config:
            kps, kds = np.full(12, 20.0), np.full(12, 0.5)
            tau_limit = np.array([23.7, 23.7, 35.55] * 4)
            default_dof_pos = np.array([0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5])

    policy = torch.jit.load(args.load_model)
    run_mujoco(policy, Sim2simCfg())
