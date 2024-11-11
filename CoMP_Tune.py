import numpy as np
from ray import tune
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gymnasium import spaces
from comyx.network import BaseStation, UserEquipment, RIS, Link
from comyx.propagation import get_noise_power
from comyx.utils import dbm2pow, generate_seed, db2pow
import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
import logging
from CoMP_MADRL import MultiAgentBaseStationEnv
from ray.tune.logger import TBXLoggerCallback  # Import TBXLoggerCallback for TensorBoard logging

logging.basicConfig(level=logging.INFO)
import os

# Set the environment variables before starting Ray
os.environ["RAY_memory_monitor_refresh_ms"] = "500"  # Check memory usage every 500ms
os.environ["RAY_memory_usage_threshold"] = "0.90"    # Start killing tasks if memory usage exceeds 90%
os.environ["RAY_TMPDIR"] = "E:\\temp"  # Or any other directory on E drive
os.environ["TMPDIR"] = "E:\\temp"  # Or any other directory on E drive

# Environment setup
Pt = np.linspace(-40, 20, 80)
Pt_lin = dbm2pow(Pt)
bandwidth = 1e6
frequency = 2.4e9
temperature = 300
mc = 1000
K = 3  # rician

N0 = get_noise_power(temperature, bandwidth)
N0_lin = dbm2pow(N0)

n_antennas = 1
n_elements = 100

nlos_fading_args = {"type": "nakagami", "m": 1, "omega": 1}
los_fading_args = {"type": "nakagami", "m": 2, "omega": 1}
bu_pathloss_args = {"type": "reference", "alpha": 3, "p0": 30, "frequency": frequency}
br_pathloss_args = {"type": "reference", "alpha": 2.7, "p0": 30, "frequency": frequency}
ru_pathloss_args = {"type": "reference", "alpha": 2.7, "p0": 30, "frequency": frequency}

BS1 = BaseStation("BS1", position=[0, 0, 30], n_antennas=n_antennas, t_power=Pt_lin)
BS2 = BaseStation("BS2", position=[0, 150, 30], n_antennas=n_antennas, t_power=Pt_lin)
BS3 = BaseStation("BS3", position=[150, 0, 30], n_antennas=n_antennas, t_power=Pt_lin)
BS4 = BaseStation("BS4", position=[150, 150, 30], n_antennas=n_antennas, t_power=Pt_lin)

UC1 = UserEquipment("UC1", position=[35.35, 35.355, 1], n_antennas=n_antennas)
UC2 = UserEquipment("UC2", position=[35.35, 185.35, 1], n_antennas=n_antennas)
UC3 = UserEquipment("UC3", position=[185.35, 35.35, 1], n_antennas=n_antennas)
UC4 = UserEquipment("UC4", position=[185.35, 185.35, 1], n_antennas=n_antennas)

R1 = RIS("RIS1", position=[53.03, 53.03, 15], n_elements=n_elements)
R2 = RIS("RIS2", position=[53.03, 203.03, 15], n_elements=n_elements)
R3 = RIS("RIS3", position=[203.03, 53.03, 15], n_elements=n_elements)
R4 = RIS("RIS4", position=[203.03, 203.03, 15], n_elements=n_elements)

UE = UserEquipment("UE", position=[75, 75, 1], n_antennas=n_antennas)

links = []
for i, bs in enumerate([BS1, BS2, BS3, BS4]):
    uc = [UC1, UC2, UC3, UC4][i]
    ris = [R1, R2, R3, R4][i]

    link_bs_uc = Link(bs, uc, nlos_fading_args, bu_pathloss_args, shape=(1, 1, mc), seed=generate_seed(f"{bs.id}-{uc.id}"))
    link_bs_ris = Link(bs, ris, los_fading_args, br_pathloss_args, shape=(n_elements, 1, mc), seed=generate_seed(f"{bs.id}-{ris.id}"), rician_args={"K": db2pow(K), "order": "post"})
    link_ris_uc = Link(ris, uc, los_fading_args, ru_pathloss_args, shape=(n_elements, 1, mc), seed=generate_seed(f"{ris.id}-{uc.id}"), rician_args={"K": db2pow(K), "order": "pre"})
    link_bs_ue = Link(bs, UE, nlos_fading_args, bu_pathloss_args, shape=(1, 1, mc), seed=generate_seed(f"{bs.id}-{UE.id}"))
    link_ris_ue = Link(ris, UE, los_fading_args, ru_pathloss_args, shape=(n_elements, 1, mc), seed=generate_seed(f"{ris.id}-{UE.id}"), rician_args={"K": db2pow(K), "order": "pre"})

    links.append((link_bs_uc, link_bs_ris, link_ris_uc, link_bs_ue, link_ris_ue))

shape_ris = (n_elements, mc)

R1.phase_shifts = np.random.uniform(0, 2 * np.pi, shape_ris)
R2.phase_shifts = np.random.uniform(0, 2 * np.pi, shape_ris)
R3.phase_shifts = np.random.uniform(0, 2 * np.pi, shape_ris)
R4.phase_shifts = np.random.uniform(0, 2 * np.pi, shape_ris)
R1.amplitudes = np.ones(shape_ris)
R2.amplitudes = np.ones(shape_ris)
R3.amplitudes = np.ones(shape_ris)
R4.amplitudes = np.ones(shape_ris)

BS_list = [BS1, BS2, BS3, BS4]
RIS_list = [R1, R2, R3, R4]
UC_list = [UC1, UC2, UC3, UC4]
threshold_rate = 1.2
comp_indices = [0]

def env_creator(env_config):
    return MultiAgentBaseStationEnv(env_config)

register_env("MultiAgentBaseStationEnv", env_creator)

env_config = {
    "BS_list": BS_list,
    "RIS_list": RIS_list,
    "UC_list": UC_list,
    "UE": UE,
    "links": links,
    "mc": mc,
    "comp_indices": comp_indices,
    "threshold_edge": threshold_rate
}

# Configure PPO with suggested changes
config = PPOConfig().copy()
config.environment(env="MultiAgentBaseStationEnv", env_config=env_config)
config.num_workers = 4
config.framework = 'torch'
config.train_batch_size = 4000
config.rollout_fragment_length = 200
config.sgd_minibatch_size = 200
config.resources(num_gpus=1)

# Learning rate, entropy, and other updates
config.lr = 3e-5  # Reduced initial learning rate
config["lr_schedule"] = [
    (0, 3e-5),
    (500000, 1e-5),
    (1000000, 3e-6)
]  # Slower decay, longer schedule
config["grad_clip"] = 0.5  # Loosen gradient clipping
config["normalize_observations"] = True
config['vf_clip_param'] = 10
#Entropy and GAE
config["entropy_coeff"] = 0.003  # Further reduced
config["entropy_coeff_schedule"] = [
    (0, 0.003),
    (1000000, 0.0005)
]  # Slower decay
config["lambda"] = 0.95  # GAE parameter
# Exploration settings for CoMP policy (Epsilon-Greedy)

# PPO-specific parameters
config["clip_param"] = 0.2
config["kl_coeff"] = 0.5  # Increased from previous suggestion
config["kl_target"] = 0.01

exploration_config_comp_policy = {
    "type": "EpsilonGreedy",
    "initial_epsilon": 1.0,
    "final_epsilon": 0.05,  # More exploration retained
    "epsilon_timesteps": 500000  # Even slower decay
}

def policy_mapping_fn(agent_id, *args, **kwargs):
    return "comp_policy" if agent_id == "CoMP" else "bs_policy"
# Gaussian noise for bs_policy
env_instance = env_creator(env_config)
config.multi_agent(
    policies={
        "bs_policy": (None, env_instance.observation_space["BS1"], env_instance.action_space["BS1"], {
            "exploration_config": {
                "type": "GaussianNoise",
                "stddev": 0.05,  # Increased stddev for more exploration
                "initial_scale": 0.5,
                "final_scale": 0.002,  # Lower final scale for exploitation
                "scale_timesteps": 200000  # Slower noise decay
            }
        }),
        "comp_policy": (None, env_instance.observation_space["CoMP"], env_instance.action_space["CoMP"], {
            "exploration_config": exploration_config_comp_policy
        }),
    },
    policy_mapping_fn=policy_mapping_fn
)
# Additional stability measures
config["num_sgd_iter"] = 20  # Increased from default
config["shuffle_sequences"] = True
config["_disable_preprocessor_api"] = False

config.reuse_actors = True
#checkpoint_path = "E:\\Projects\\Unfinished\\NOMA\\Links\\No Abstraction\\Comyx\\MADRL\\Rate Threshold Pathloss 3\\Threshold 1.4\\PPO_2024-09-27_00-20-17\\PPO_MultiAgentBaseStationEnv_5d596_00000_0_2024-09-27_00-20-18\\checkpoint_000000"

# Run the training process
analysis = tune.run(
    "PPO",
    config=config.to_dict(),
    stop={"training_iteration": 200},  # Increased training iterations
    checkpoint_freq=199,  # Save checkpoints every 50 iterations
    storage_path="E:\\Projects\\Unfinished\\NOMA\\Links\\No Abstraction\\Comyx\\MADRL\\Rate Maximized",
    metric="env_runners/episode_reward_mean",
    mode="max",  # Assuming you want to maximize the mean reward
    verbose=1,
    callbacks=[TBXLoggerCallback()]  # TensorBoard logging
)

# Save the best checkpoint
best_checkpoint = analysis.best_checkpoint
print(f"Best checkpoint: {best_checkpoint}")

# To track rewards over each iteration, start TensorBoard:
# tensorboard --logdir=./ray_results/
