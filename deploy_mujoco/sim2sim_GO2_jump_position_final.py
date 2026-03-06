from legged_gym.envs.Go2_MoB.GO2_JUMP.GO2_JUMP_position_config import GO2_JUMP_Position_Cfg
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
current_speed_target = 1.0 
key_states = {'w': False, 's': False, 'a': False, 'd': False, 'q': False, 'e': False}

def on_press(key):
    global current_speed_target
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

def get_joint_mapping(model):
    '''根据名称对齐 Isaac Gym 的关节顺序和执行器顺序'''
    isaac_joint_names = [
        'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
        'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
        'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint',
        'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint'
    ]
    qpos_ids, qvel_ids, actuator_ids = [], [], []
    for name in isaac_joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid == -1: raise ValueError(f"Joint {name} not found!")
        qpos_ids.append(model.jnt_qposadr[jid])
        qvel_ids.append(model.jnt_dofadr[jid])
        found_actuator = False
        for aid in range(model.nu):
            if model.actuator_trntype[aid] == mujoco.mjtTrn.mjTRN_JOINT and model.actuator_trnid[aid, 0] == jid:
                actuator_ids.append(aid)
                found_actuator = True
                break
        if not found_actuator: raise ValueError(f"No actuator for {name}!")
    return np.array(qpos_ids), np.array(qvel_ids), np.array(actuator_ids)

def pd_control(target_q, q, kp, dq, kd, default_q):
    '''位置控制专用的 PD 计算'''
    return kp * (target_q + default_q - q) - kd * dq

def run_mujoco(policy, cfg, args):
    global x_vel_cmd, y_vel_cmd, yaw_vel_cmd
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    
    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    model.opt.timestep = cfg.sim_config.dt
    
    # 负载注入逻辑
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    if base_id == -1: base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    original_mass = model.body_mass[base_id]
    if args.load_mass > 0:
        model.body_mass[base_id] = original_mass + args.load_mass
        print(f"\n>>> [POS-MODEL] Payload Applied: +{args.load_mass}kg (Total: {model.body_mass[base_id]:.2f}kg) <<<")
        for gid in range(model.ngeom):
            if model.geom_bodyid[gid] == base_id: model.geom_rgba[gid] = [1, 0, 0, 1]

    data = mujoco.MjData(model)
    q_ids, dq_ids, act_ids = get_joint_mapping(model)
    data.qpos[q_ids] = cfg.robot_config.default_dof_pos
    mujoco.mj_step(model, data)
    
    count_lowlevel = 0
    frame_stack = cfg.env.frame_stack
    hist_obs = deque(maxlen=frame_stack)
    for _ in range(frame_stack):
        hist_obs.append(np.zeros([1, cfg.env.num_single_obs], dtype=np.double))
    
    action = np.zeros(cfg.env.num_actions, dtype=np.double)
    max_vel_reached = 0.0
    total_abs_error_x = 0.0
    tracking_steps = 0

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

            q, dq = data.qpos[q_ids].astype(np.double), data.qvel[dq_ids].astype(np.double)
            q_raw = data.qpos[3:7].astype(np.double) 
            quat_scipy = q_raw[[1, 2, 3, 0]] 
            omega = data.qvel[3:6].astype(np.double) 
            eu_ang = quaternion_to_euler_array(quat_scipy)
            eu_ang[eu_ang > math.pi] -= 2 * math.pi
            
            # 50Hz 策略推理
            if count_lowlevel % cfg.sim_config.decimation == 0:
                target_vx = (1.0 if key_states['w'] else 0.0) - (1.0 if key_states['s'] else 0.0)
                target_vy = (1.0 if key_states['a'] else 0.0) - (1.0 if key_states['d'] else 0.0)
                target_vyaw = (1.0 if key_states['q'] else 0.0) - (1.0 if key_states['e'] else 0.0)
                x_vel_cmd = x_vel_cmd * 0.95 + target_vx * current_speed_target * 0.05
                y_vel_cmd = y_vel_cmd * 0.95 + target_vy * current_speed_target * 0.05
                yaw_vel_cmd = yaw_vel_cmd * 0.95 + target_vyaw * current_speed_target * 0.05
                
                obs = np.zeros([1, cfg.env.num_single_obs], dtype=np.float32)
                phase = (current_sim_time) % cfg.rewards.cycle_time / cfg.rewards.cycle_time
                obs[0, 0], obs[0, 1] = math.sin(2 * math.pi * phase), math.cos(2 * math.pi * phase)
                obs[0, 2], obs[0, 3], obs[0, 4] = x_vel_cmd * 2.0, y_vel_cmd * 2.0, yaw_vel_cmd * 0.25
                obs[0, 5:11] = np.concatenate([omega * 0.25, eu_ang * 1.0])
                obs[0, 11:35] = np.concatenate([(q - cfg.robot_config.default_dof_pos) * 1.0, dq * 0.05])
                obs[0, 35:47] = action   

                hist_obs.append(np.clip(obs, -100., 100.))
                policy_input = np.zeros([1, cfg.env.num_observations], dtype=np.float32)
                for i in range(frame_stack):
                    policy_input[0, i*47 : (i+1)*47] = hist_obs[i][0, :]

                action[:] = policy(torch.tensor(policy_input))[0].detach().numpy()
                action = np.clip(action, -100., 100.)
                viewer.sync()

            # 200Hz PD 控制
            target_q_scaled = action * cfg.control.action_scale
            tau = pd_control(target_q_scaled, q, cfg.robot_config.kps, dq, cfg.robot_config.kds, cfg.robot_config.default_dof_pos)
            tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)
            data.ctrl[act_ids] = tau

            # 指标记录
            actual_vel = np.linalg.norm(data.qvel[:2])
            if actual_vel > max_vel_reached: max_vel_reached = actual_vel
            if current_sim_time > 2.0 and np.abs(x_vel_cmd) > 0.5:
                total_abs_error_x += np.abs(x_vel_cmd - data.qvel[0]); tracking_steps += 1
            
            if count_lowlevel % 20 == 0:
                mae = total_abs_error_x / max(tracking_steps, 1)
                print(f"\r[POS-MODE] V: {actual_vel:.2f} | MAE: {mae:.3f} | Hz: {real_hz:.1f} ", end="")

            mujoco.mj_step(model, data)
            count_lowlevel += 1
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0: time.sleep(time_until_next_step)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_model', type=str, required=True)
    parser.add_argument('--terrain', action='store_true')
    parser.add_argument('--load_mass', type=float, default=0.0)
    args = parser.parse_args()

    class Sim2simCfg(GO2_JUMP_Position_Cfg):
        class sim_config:
            mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/go2/scene{"_terrain" if args.terrain else ""}.xml'
            dt, decimation = 0.005, 4
        class robot_config:
            kps, kds = np.full(12, 20.0), np.full(12, 0.5)
            tau_limit = 23.7 * np.ones(12) # 严格物理限幅
            default_dof_pos = np.array([0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5])

    policy = torch.jit.load(args.load_model)
    run_mujoco(policy, Sim2simCfg(), args)
