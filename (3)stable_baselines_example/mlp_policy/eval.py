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

#path = "./mlp_policy/PPO_6/jetbot_policy"
# best so far
path = "./mlp_policy/PPO_gamma/PPO_0.99975/"
#policy = "jetbot_policy_checkpoint_500000_steps" 
#path = "./mlp_policy/PPO_0.99975/"
policy = "jetbot_policy_checkpoint_500000_steps" 

policy_path = path + policy
my_env = JetBotEnv(headless=False)
model = PPO.load(policy_path)

trials = 20
successes = 0
collisions = 0
time_limit = 0

for _ in range(trials):
    obs = my_env.reset()
    done = False
    while not done:
        actions, _ = model.predict(observation=obs, deterministic=False)
        obs, reward, done, info = my_env.step(actions)
        if info["Successes"] == True:
            successes += 1
        if info["Collisions"] == True:
            collisions += 1
        if info["Time Limit"] == True:
            time_limit += 1
        my_env.render()

print("Total successes out of 20: ", successes)

score = successes / trials

filename = path + 'eval.txt'
with open(filename, 'w') as f:
    f.write("Policy: " + str(policy_path))
    f.write("\nTotal trials: " + str(trials))
    f.write("\nTotal collisions: " + str(collisions))
    f.write("\nTime exceeded: " + str(time_limit))
    f.write("\nTotal successes: " + str(successes))
    f.write("\nPerformance score: " + str(score))

my_env.close()
