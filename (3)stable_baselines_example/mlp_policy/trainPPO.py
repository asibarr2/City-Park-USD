# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

from testenv import JetBotEnv
from stable_baselines3 import PPO
from stable_baselines3.ppo import CnnPolicy
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList
import torch as th
import numpy as np
import shutil

log_dir = "./mlp_policy/Demo"
my_env = JetBotEnv(headless=False) # set headless to false to visualize training

policy_kwargs = dict(activation_fn=th.nn.Tanh, net_arch=[16, dict(pi=[64, 32], vf=[64, 32])])
policy = CnnPolicy
total_timesteps = 500000

checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=log_dir, name_prefix="jetbot_policy_checkpoint")

model = PPO(policy, my_env, policy_kwargs=policy_kwargs, verbose=1,
    n_steps=10000,
    batch_size=1000,
    learning_rate=0.00025,
    gamma=0.99975,
    device="cuda",
    ent_coef=0.1,
    vf_coef=0.5,
    max_grad_norm=10,
    tensorboard_log=log_dir,
)

"""
trainFile = r'trainPPO.py'
envFile = r'testenv.py'
trainParams = r'mlp_policy/Demo/parameters.txt'
envParams = r'mlp_policy/Demo/testenv.txt'
shutil.copyfile(trainFile, trainParams)
shutil.copyfile(envFile, envParams)
"""

model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback])
model.save(log_dir + "/jetbot_policy")

my_env.close()
