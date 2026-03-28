import sys
import os
import numpy as np
import isaacgym
import torch
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger

# --- 配置开关 ---
EXPORT_POLICY = True
RECORD_FRAMES = False
MOVE_CAMERA = True
# ----------------

def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    
    # 强制单机模式，聚焦观察
    env_cfg.env.num_envs = 1
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.commands.resampling_time = 1e9 
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.push_towards_goal = False

    env_cfg.env.test = True

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    
    # 同步 PD 衰减进度
    if args.checkpoint != -1:
        try:
            iter_num = int(args.checkpoint)
            steps_per_iter = train_cfg.runner.num_steps_per_env
            env.common_step_counter = iter_num * steps_per_iter
            print(f"[Play] Syncing curriculum: Set common_step_counter to {env.common_step_counter}")
        except:
            print("[Play] Warning: Could not parse checkpoint.")
            
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)
    
    robot_index = 0 
    img_idx = 0
    
    # 设定目标对标速度
    target_vel = 3.2
    env.commands[:, 0] = target_vel
    env.commands[:, 1] = 0.
    env.commands[:, 2] = 0.
    
    print(f"\n[Play] Running at Target Velocity: {target_vel} m/s")

    for i in range(10 * int(env.max_episode_length)):
        env.commands[:, 0] = target_vel
        
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())
        
        # 视角跟随
        if MOVE_CAMERA and not env.headless:
            robot_pos = env.root_states[robot_index, :3].cpu().numpy()
            cam_offset = np.array([-2.5, 1.5, 1.0])
            camera_position = robot_pos + cam_offset
            env.set_camera(camera_position, robot_pos)

        # 每 100 步输出一次简报
        if i % 100 == 0:
            actual_vel = env.base_lin_vel[robot_index, 0].item()
            max_torque = torch.max(torch.abs(env.torques[robot_index])).item()
            print(f"Step {i:5d} | Cmd: {target_vel:.2f} | Actual: {actual_vel:.3f} | Max Torque: {max_torque:.2f} Nm")

if __name__ == '__main__':
    args = get_args()
    play(args)
