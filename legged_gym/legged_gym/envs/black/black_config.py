# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from legged_gym import LEGGED_GYM_ROOT_DIR

class BlackRoughCfg( LeggedRobotCfg ):
    class env(LeggedRobotCfg.env):
        num_envs = 2048
    class terrain( LeggedRobotCfg.terrain ):
        # mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        mesh_type = 'trimesh'
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete,steping stone, gap, pit]
        # terrain_proportions = [0.1, 0.4, 0.2, 0.2, 0.1]
        # terrain_proportions = [0.1, 0.2, 0.3, 0.3, 0.1]
        # terrain_proportions = [0.1, 0.1, 0.5, 0.3, 0.0 ,0.0, 0.0, 0.0]
    
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.42] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1 ,  # [rad]
            'FR_hip_joint': 0.1 ,  # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'RR_hip_joint': 0.1,   # [rad]

            'FL_thigh_joint': -0.8,     # [rad]
            'FR_thigh_joint': -0.8,     # [rad]
            'RL_thigh_joint': -0.8,   # [rad]
            'RR_thigh_joint': -0.8,   # [rad]

            'FL_calf_joint': 1.4,   # [rad]
            'FR_calf_joint': 1.4,  # [rad]
            'RL_calf_joint': 1.4,    # [rad]
            'RR_calf_joint': 1.4,    # [rad]
            # 'FL_hip_joint': 0.3 ,  # [rad]
            # 'FR_hip_joint': 0.3 ,  # [rad]
            # 'RL_hip_joint': 0.3,   # [rad]
            # 'RR_hip_joint': 0.3,   # [rad]

            # 'FL_thigh_joint': -0.4,     # [rad]
            # 'FR_thigh_joint': -0.4,     # [rad]
            # 'RL_thigh_joint': -0.4,   # [rad]
            # 'RR_thigh_joint': -0.4,   # [rad]

            # 'FL_calf_joint': 0.6,   # [rad]
            # 'FR_calf_joint': 0.6,  # [rad]
            # 'RL_calf_joint': 0.6,    # [rad]
            # 'RR_calf_joint': 0.6,    # [rad]
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 40.0}  # [N*m/rad] 28
        damping = {'joint': 1.0}     # [N*m*s/rad] 0.7
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        hip_reduction = 0.5
    

    class commands( LeggedRobotCfg.commands ):
        command_curriculum = True
        curriculum = False
        curriculum_seed = 100
        max_curriculum = 2
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        num_lin_vel_bins = 21
        num_y_vel_bins = 1
        num_yaw_vel_bins = 21
        resampling_time = 10. # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        limit_lin_vel_x = [-1.0, 1.0]
        limit_lin_vel_y = [-1.0, 1.0]
        limit_ang_vel_yaw = [-3.14, 3.14]
        local_range = 0.5
        class ranges( LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-1.0, 1.0] # min max [m/s]
            lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            ang_vel_yaw = [-3.14, 3.14]    # min max [rad/s]
            heading = [-3.14, 3.14]
           

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/black/black.urdf'
        name = "black"
        foot_name = "toe_link"
        penalize_contacts_on = ["thigh", "calf", "base"]
        terminate_after_contacts_on = ["base"]
        privileged_contacts_on = ["base", "thigh", "calf"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False # Some .obj meshes must be flipped from y-up to z-up
        collapse_fixed_joints = True

    # class domain_rand( LeggedRobotCfg.domain_rand):
    #     randomize_restitution = True # TODO 添加弹性恢复系数域随机
    #     restitution_range = [0., 0.2]
    
    class rewards( LeggedRobotCfg.rewards ):
        class scales:
            # 默认参数
            termination = -0.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -0.2
            dof_acc = -2.5e-7
            joint_power = -2e-5
            base_height = -1.0
            foot_clearance = -0.0
            action_rate = -0.01
            smoothness = -0.01
            feet_air_time =  0.0
            collision = -0.01
            feet_stumble = -0.0       
            stand_still = -0.0
            torques = -0.0
            dof_vel = -0.0
            dof_pos_limits = -0.0 # TODO 惩罚超限
            dof_vel_limits = -0.0
            torque_limits = -0.0



        # TODO
        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized TODO
        soft_dof_vel_limit = 0.99
        soft_torque_limit = 0.99
        base_height_target = 0.32 #0.32
        max_contact_force = 100. # forces above this value are penalized
        clearance_height_target = -0.20

class BlackRoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'rough_black'

        resume = False
        load_run =  '/home/guo/STU/HIMLoco/policies' 
        checkpoint = -1 # -1 = last saved model
        save_interval = 2000 # check for potential saves every this many iterations
        max_iterations = 7000



# /home/guo/STU/HIMLoco/legged_gym/logs/rough_black/Mar04_09-32-17_ 
# /home/guo/STU/HIMLoco/legged_gym/logs/rough_black/Mar03_15-16-10_
# /home/guo/STU/HIMLoco/legged_gym/logs/rough_black/Feb02_22-39-15_