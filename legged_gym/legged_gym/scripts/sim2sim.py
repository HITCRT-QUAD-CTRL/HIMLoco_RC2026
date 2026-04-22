import math
import numpy as np
import mujoco, mujoco.viewer
from tqdm import tqdm
import argparse
from collections import deque
from scipy.spatial.transform import Rotation as R
from legged_gym import LEGGED_GYM_ROOT_DIR
import pygame


# from legged_gym.envs.go1.go1_config import WhiteRoughCfg
from legged_gym.envs.white.white_config import WhiteRoughCfg
import torch

asset_dof_names = [
                   'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
                   'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
                   'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint',
                   'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint']
                   

# asset_dof_names = ['LF_hip_joint', 'LF_thigh_joint', 'LF_calf_joint',
#                    'RF_hip_joint', 'RF_thigh_joint', 'RF_calf_joint',
#                    'LB_hip_joint', 'LB_thigh_joint', 'LB_calf_joint',
#                    'RB_hip_joint', 'RB_thigh_joint', 'RB_calf_joint']

default_dof_pos = np.zeros(12)
for i in range(len(asset_dof_names)):
    default_dof_pos[i] = WhiteRoughCfg.init_state.default_joint_angles[asset_dof_names[i]]

class cmd:
    vx = 1.0
    vy = 0.0
    dyaw = 0.0

# 手柄控制相关变量
joystick = None
pygame_initialized = False

def init_joystick():
    """初始化pygame和手柄"""
    global joystick, pygame_initialized
    if not pygame_initialized:
        pygame.init()
        pygame_initialized = True
        pygame.joystick.init()
        
        # 检查是否有手柄连接
        if pygame.joystick.get_count() > 0:
            joystick = pygame.joystick.Joystick(0)
            joystick.init()
            print(f"检测到手柄: {joystick.get_name()}")
            print(f"轴数量: {joystick.get_numaxes()}")
            print(f"按钮数量: {joystick.get_numbuttons()}")
        else:
            print("未检测到手柄，将使用默认命令")

def update_cmd_from_joystick():
    """从手柄读取数据并更新命令
    
    手柄与速度映射:
    lin_x = -ly
    lin_y = -lx
    ang_yaw = -rx
    """
    global joystick
    if joystick is None:
        return
    
    # 处理pygame事件
    for event in pygame.event.get():
        pass  # 处理事件队列
    
    # 读取手柄轴数据
    # 假设: 轴0是左摇杆X(lx), 轴1是左摇杆Y(ly), 轴3是右摇杆X(rx)
    # 不同手柄可能有不同的轴映射，需要根据实际情况调整
    lx = joystick.get_axis(0)
    ly = joystick.get_axis(1)
    rx = joystick.get_axis(2)  # 右摇杆X轴
    
    # 应用手柄到速度的映射
    cmd.vx = -ly
    cmd.vy = -lx
    cmd.dyaw = -rx

def low_pass_action_filter(actions, last_actions):
  alpha = 0.2
  actons_filtered = last_actions * alpha + actions * (1 - alpha)
  return actons_filtered

def get_obs(data):
    # q = data.qpos.astype(np.double)
    # dq = data.qvel.astype(np.double)
    # q = q[-cfg.env.num_actions:]
    # dq = dq[-cfg.env.num_actions:]
    q = np.zeros(12)
    dq = np.zeros(12)
    for i in range(12):
        q[i] = data.joint(asset_dof_names[i]).qpos[0]
        dq[i] = data.joint(asset_dof_names[i]).qvel[0]

    # quat = data.sensor('orientation').data[[1, 2, 3, 0]].astype(np.double)
    quat = data.sensor('imu_quat').data[[1, 2, 3, 0]].astype(np.double)
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # In the base frame
    # omega = data.sensor('angular-velocity').data.astype(np.double)
    omega = data.sensor('imu_angvel').data.astype(np.double)
    gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
    return (q, dq, quat, v, omega, gvec)

def pd_control(target_q, q, kp, target_dq, dq, kd):
    '''
    Calculates torques from position commands
    '''
    return (target_q - q) * kp + (target_dq - dq) * kd

def run_mujoco(policy, cfg):
    global default_dof_pos
    # 初始化手柄
    init_joystick()
    
    model = mujoco.MjModel.from_xml_path(cfg.mjsim_config.mujoco_model_path) # 载入初始化位置由XML决定
    model.opt.timestep = cfg.mjsim_config.dt
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    viewer = mujoco.viewer.launch_passive(model, data)

    target_q = np.zeros((cfg.env.num_actions), dtype=np.double)
    target_dq = np.zeros((cfg.env.num_actions), dtype=np.double)
    action = np.zeros((cfg.env.num_actions), dtype=np.double)
    command = np.zeros((1, 3), dtype=np.double)
    last_action = action[:]
    hist_obs_vec = np.zeros([1, 6 * cfg.env.num_one_step_observations], dtype=np.double)
    
    hist_obs_deque = deque()
    for _ in range(6):
        hist_obs_deque.append(np.zeros([1, cfg.env.num_one_step_observations], dtype=np.double))
    
    count_lowlevel = 0

    for _ in tqdm(range(int(cfg.mjsim_config.sim_duration / cfg.mjsim_config.dt)), desc="Simulating..."):

        # Obtain an observation
        q, dq, quat, v, omega, gvec = get_obs(data) #从mujoco获取仿真数据
        if True:
            # 1000hz ->50hz
            if count_lowlevel % cfg.mjsim_config.decimation == 0:
                obs = np.zeros([1, cfg.env.num_one_step_observations], dtype=np.double) # 1,45
                # ---- set command ------
                # 从手柄读取命令
                update_cmd_from_joystick()
                # ---- set command ------
                
                # current_obs = torch.cat((self.commands[:, :3] * self.commands_scale,
                #             self.base_ang_vel  * self.obs_scales.ang_vel,
                #             self.projected_gravity,
                #             (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                #             self.dof_vel * self.obs_scales.dof_vel,
                #             self.actions
                #             ),dim=-1)
                
                obs[0, 0] = cmd.vx * cfg.normalization.obs_scales.lin_vel
                obs[0, 1] = cmd.vy * cfg.normalization.obs_scales.lin_vel
                obs[0, 2] = cmd.dyaw * cfg.normalization.obs_scales.ang_vel
                obs[0, 3] = omega[0] * cfg.normalization.obs_scales.ang_vel
                obs[0, 4] = omega[1] * cfg.normalization.obs_scales.ang_vel
                obs[0, 5] = omega[2] * cfg.normalization.obs_scales.ang_vel
                obs[0, 6] = gvec[0]
                obs[0, 7] = gvec[1]
                obs[0, 8] = gvec[2]
                obs[0, 9:21] = (q - default_dof_pos) * cfg.normalization.obs_scales.dof_pos
                obs[0, 21:33] = dq * cfg.normalization.obs_scales.dof_vel
                obs[0, 33:45] = last_action
            
                hist_obs_deque.append(obs)
                hist_obs_deque.popleft()
                
                # if(count_lowlevel%1000==0):
                #     print(obs)

                #-------缓存历史观测作为模型输入---
                for i in range(6):#缓存历史观测
                    hist_obs_vec[0, i * cfg.env.num_one_step_observations : (i + 1) * cfg.env.num_one_step_observations] = hist_obs_deque[6 - 1 - i][0, :]
                    # hist_obs_vec[0, i * cfg.env.num_one_step_observations : (i + 1) * cfg.env.num_one_step_observations] = hist_obs_deque[i][0, :]
                    
                
                #-------推理jit save的模型-------
                policy = policy.to('cpu')
                action[:] = policy(torch.tensor(hist_obs_vec).to(torch.float32))[0].detach().numpy()#jit模型

                #-------计算力矩输出-------------
                action = np.clip(action, -cfg.normalization.clip_actions, cfg.normalization.clip_actions)
                # action_flt = 0.8 * action + 0.2 * last_action
                last_action = action
                # last_action = action_flt

                # for a in action:
                #     if(a>10):
                #         print(action)

                #-------计算前馈力矩-------------
                actions_scaled = last_action * cfg.control.action_scale
                actions_scaled[[0, 3, 6, 9]] *= cfg.control.hip_reduction
                target_q = actions_scaled + default_dof_pos 
            
            if cfg.control.control_type=="P":
                torques = (target_q - q) * cfg.robot_config.kps - dq * cfg.robot_config.kds
            else:
                raise NameError(f"Unknown controller type: {cfg.control.control_type}")
    
            data.ctrl = torques    
            # torques = np.clip (torques, - cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)
           
        else: # air mode test
            target_q = default_dof_pos
            target_dq = np.zeros((cfg.env.num_actions), dtype=np.double)
            # Generate PD control
            tau = (target_q - q) * cfg.robot_config.kps + (target_dq - dq) * cfg.robot_config.kds
            tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)  # Clamp torques
            data.ctrl = tau # np.zeros((cfg.env.num_actions), dtype=np.double)
            
            # for i in range(sim.data.ncon):
            #     con = sim.data.contact[i]
            #     if (con.geom1 == geom1_id and con.geom2 == geom2_id) or (con.geom2 == geom1_id and con.geom1 == geom2_id):
            #         contact_pos = con.pos
            #         break
            
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
            body_pos = data.xpos[body_id]
            print(f"The position of the body base is: {body_pos}")
            
        mujoco.mj_step(model, data)
        if count_lowlevel % cfg.mjsim_config.decimation == 0:
            # viewer.cam.distance = 3.0
            # viewer.cam.azimuth = 90
            # viewer.cam.elevation = -45
            viewer.cam.lookat[:] = data.qpos.astype(np.double)[0:3]
            viewer.sync()
        count_lowlevel += 1

    viewer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deployment script.')
    parser.add_argument('--load_model', type=str, default='model_vqvae.pt',
                        help='Run to load from.')
    parser.add_argument('--terrain', action='store_true', default=False)
    args = parser.parse_args()
    
    args.terrain = True
    # args.load_model = f'{LEGGED_GYM_ROOT_DIR}/logs/rough_white/exported/policies/policy.pt'
    args.load_model = f'/home/guo/HITCRT_QUAD/modules/model_converter/onnx_converter/exported/jit_model.pt'
    class Sim2simCfg(WhiteRoughCfg):
        class mjsim_config:
            if args.terrain:
                # mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/go1/xml/world_terrain.xml'
                # mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/go1/xml/world_stairs.xml'
                mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/white/white.xml'
            else:
                mujoco_model_path = f'{LEGGED_GYM_ROOT_DIR}/resources/robots/go1/xml/world.xml'
            sim_duration = 600.0
            dt = 0.005
            decimation = 4 #50Hz

        class robot_config:
            kp_all = WhiteRoughCfg.control.stiffness['joint'] 
            kd_all = WhiteRoughCfg.control.damping['joint'] 
            kp_all = 32 
            kd_all = 0.8
            
            kps = np.array([kp_all, kp_all, kp_all, kp_all, kp_all, kp_all, kp_all, kp_all, kp_all, kp_all, kp_all, kp_all], dtype=np.double)
            kds = np.array([kd_all, kd_all, kd_all, kd_all, kd_all, kd_all, kd_all, kd_all, kd_all, kd_all, kd_all, kd_all], dtype=np.double)
            # tau_limit = np.array([20., 55., 55., 20., 55., 55., 20., 55., 55., 20., 55., 55.])
            tau_limit = np.array([23.7, 23.7, 23.7, 23.7, 23.7, 23.7, 23.7, 23.7, 23.7, 23.7, 23.7, 23.7])
        class env(WhiteRoughCfg.env):
            pass
            
        class normalization(WhiteRoughCfg.normalization):
            pass            

    policy = torch.jit.load(args.load_model)
    policy.eval() 

    run_mujoco(policy, Sim2simCfg())
