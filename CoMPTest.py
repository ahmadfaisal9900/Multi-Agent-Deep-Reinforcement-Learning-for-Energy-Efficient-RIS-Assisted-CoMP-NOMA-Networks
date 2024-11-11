import numpy as np
import ray
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.tune.registry import register_env
from CoMP_MADRL import MultiAgentBaseStationEnv
import logging
from comyx.network import BaseStation, UserEquipment, RIS, Link
from comyx.propagation import get_noise_power
from comyx.utils import dbm2pow, generate_seed, db2pow
from tqdm import tqdm

# Initialize logging and Ray
logging.basicConfig(level=logging.INFO)
ray.init(num_cpus=3, num_gpus=1)

# Define environment configuration
Pt = np.linspace(-40, 20, 80)
Pt_lin = dbm2pow(Pt)
bandwidth = 1e6
frequency = 2.4e9
temperature = 300
mc = 10
K = 3
N0 = get_noise_power(temperature, bandwidth)
N0_lin = dbm2pow(N0)
n_antennas = 1
n_elements = 100

nlos_fading_args = {"type": "nakagami", "m": 1, "omega": 1}
los_fading_args = {"type": "nakagami", "m": 2, "omega": 1}
#Alpha changed from 3 to 4.5
#Alpha also changed from 2.7 to 4.2
bu_pathloss_args = {"type": "reference", "alpha": 3, "p0": 30, "frequency": frequency}
br_pathloss_args = {"type": "reference", "alpha": 2.7, "p0": 30, "frequency": frequency}
ru_pathloss_args = {"type": "reference", "alpha": 2.7, "p0": 30, "frequency": frequency}

BS1 = BaseStation("BS1", position=[0, 0, 30], n_antennas=n_antennas, t_power=Pt_lin)
BS2 = BaseStation("BS2", position=[75, 0, 30], n_antennas=n_antennas, t_power=Pt_lin)
BS3 = BaseStation("BS3", position=[0, 75, 30], n_antennas=n_antennas, t_power=Pt_lin)
BS4 = BaseStation("BS4", position=[75, 75, 30], n_antennas=n_antennas, t_power=Pt_lin)

UC1 = UserEquipment("UC1", position=[50, 0, 1], n_antennas=n_antennas)
UC2 = UserEquipment("UC2", position=[0, 50, 1], n_antennas=n_antennas)
UC3 = UserEquipment("UC3", position=[35.36, 35.36, 1], n_antennas=n_antennas)
UC4 = UserEquipment("UC4", position=[60, 60, 1], n_antennas=n_antennas)

R1 = RIS("RIS1", position=[75, 0, 10], n_elements=n_elements)
R2 = RIS("RIS2", position=[0, 75, 10], n_elements=n_elements)
R3 = RIS("RIS3", position=[53, 53, 10], n_elements=n_elements)
R4 = RIS("RIS4", position=[65, 65, 10], n_elements=n_elements)

UE = UserEquipment("UE", position=[150, 15, 1], n_antennas=n_antennas)

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
comp_indices = [0, 1]

env_config = {
    "BS_list": BS_list,
    "RIS_list": RIS_list,
    "UC_list": UC_list,
    "UE": UE,
    "links": links,
    "mc": mc,
    "comp_indices": comp_indices,
    "threshold_edge": threshold_rate,
    "testing_mode": True  # Activate testing mode
}

# Register the environment
def env_creator(env_config):
    return MultiAgentBaseStationEnv(env_config)

register_env("MultiAgentBaseStationEnv", env_creator)

# Create the environment instance for testing
test_env_instance = MultiAgentBaseStationEnv(env_config)
# Original path with forward slashes
original_path = 'E:/Projects/Unfinished/NOMA/Links/No Abstraction/Comyx/MADRL/Rate Maximized/PPO_2024-10-06_19-46-25/PPO_MultiAgentBaseStationEnv_c3193_00000_0_2024-10-06_19-46-26/checkpoint_000000'

# # Replace forward slashes with backward slashes
modified_path = original_path.replace("/", "\\")
best_checkpoint_path = modified_path
#best_checkpoint_path = original_path
# Load the algorithm with the best checkpoint
config = PPOConfig().environment("MultiAgentBaseStationEnv", env_config=env_config)
config = config.framework('torch')

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

algorithm = PPO(config=config)
def evaluate_single_checkpoint(algorithm, env, checkpoint_path, num_steps=200, num_channel_realizations=1): 
    env.efficiency_threshold = 100000

    # Load the corresponding checkpoint
    algorithm.restore(checkpoint_path)
    
    original_edge_rates = []
    original_near_rates = []
    original_near_pre_rates = []
    final_edge_rates = []
    final_near_rates = []
    final_near_pre_rates = []
    
    comp_index_list_before = []  # List to store CoMP indices before optimization
    comp_index_list_after = []  # List to store CoMP indices after optimization
    
    original_efficiency_list = []
    final_efficiency_list = []
    
    # Separate arrays to store initial and final outage rates for each realization
    outage_rates_near_all_realizations_initial = []
    outage_rates_edge_all_realizations_initial = []
    
    outage_rates_near_all_realizations_final = []
    outage_rates_edge_all_realizations_final = []

    for _ in range(num_channel_realizations):
        obs, infos = env.reset()
        
        # Collect original rates for all channel realizations
        edge_rates = []
        near_rates = []
        near_pre_rates = []
        
        outage_rates_near_realization_initial = []
        outage_rates_edge_realization_initial = []
        
        for _ in tqdm(range(env.mc), desc="Original Realizations"):
            _, _ = env.reset()
            # Get original rates
            original_edge_rate, original_near_rate, original_near_pre_rate = env.calculate_sinr_allocation_multiple(
                env.Pt_lin, env.gains_near, env.gains_edge, env.N0_lin, env.fixed_power_allocations, env.comp_indices)
            
            edge_rates.append(original_edge_rate.flatten())
            near_rates.append(original_near_rate.flatten())
            near_pre_rates.append(original_near_pre_rate.flatten())
            
            # Track the CoMP indices before optimization
            comp_index_list_before.append(env.comp_indices.copy())
            
            # Update outage probabilities (initial)
            env.update_outage_rates(original_near_rate.flatten(), original_near_pre_rate.flatten(), original_edge_rate.flatten())
            
            # Store outage rates for this realization (initial)
            outage_rates_near_realization_initial.append(env.outage_rates_near)
            outage_rates_edge_realization_initial.append(env.outage_rate_edge)
            
        # Store initial outage rates for all realizations
        outage_rates_near_all_realizations_initial.append(np.mean(outage_rates_near_realization_initial, axis=0))
        outage_rates_edge_all_realizations_initial.append(np.mean(outage_rates_edge_realization_initial, axis=0))
        
        original_edge_rates.append(edge_rates)
        original_near_rates.append(near_rates)
        original_near_pre_rates.append(near_pre_rates)
        
        # Collect final rates for all channel realizations
        edge_rates = []
        near_rates = []
        near_pre_rates = []
        
        outage_rates_near_realization_final = []
        outage_rates_edge_realization_final = []
        
        for _ in tqdm(range(env.mc), desc="Optimal Realizations"):
            obs, info = env.reset()
            
            # Run the episode
            for step in range(num_steps):
                actions = {}
                env.update_channel_realization()

                for agent_id in env.agents:
                    policy_id = "comp_policy" if agent_id == "CoMP" else "bs_policy"
                    actions[agent_id] = algorithm.compute_single_action(obs[agent_id], policy_id=policy_id)

                obs, rewards, terminated, truncated, infos = env.step(actions)
                
                if terminated["__all__"]:
                    break

            # Track the CoMP indices after optimization
            comp_index_list_after.append(env.comp_indices.copy())
            
            # Get final rates
            final_edge_rate, final_near_rate, final_near_pre_rate = env.calculate_sinr_allocation_multiple(
                env.Pt_lin, env.gains_near, env.gains_edge, env.N0_lin, env.fixed_power_allocations, env.comp_indices)

            edge_rates.append(final_edge_rate.flatten())
            near_rates.append(final_near_rate.flatten())
            near_pre_rates.append(final_near_pre_rate.flatten())
            
            # Update outage probabilities (final)
            env.update_outage_rates(final_near_rate.flatten(), final_near_pre_rate.flatten(), final_edge_rate.flatten())
            
            # Store outage rates for this realization (final)
            outage_rates_near_realization_final.append(env.outage_rates_near)
            outage_rates_edge_realization_final.append(env.outage_rate_edge)
        
        # Store final outage rates for all realizations
        outage_rates_near_all_realizations_final.append(np.mean(outage_rates_near_realization_final, axis=0))
        outage_rates_edge_all_realizations_final.append(np.mean(outage_rates_edge_realization_final, axis=0))
        
        final_edge_rates.append(edge_rates)
        final_near_rates.append(near_rates)
        final_near_pre_rates.append(near_pre_rates)
    # Convert to numpy arrays for easier manipulation
    num_channel_realizations = mc
    original_edge_rates = np.array(original_edge_rates).reshape(num_channel_realizations, 1)
    original_near_rates = np.array(original_near_rates).reshape(num_channel_realizations, 4, 1, 1)
    original_near_pre_rates = np.array(original_near_pre_rates).reshape(num_channel_realizations, 4, 1, 1)
    final_edge_rates = np.array(final_edge_rates).reshape(num_channel_realizations, 1)
    final_near_rates = np.array(final_near_rates).reshape(num_channel_realizations, 4, 1, 1)
    final_near_pre_rates = np.array(final_near_pre_rates).reshape(num_channel_realizations, 4, 1, 1)

    # Calculate the means across channel realizations (retain base station dimension)
    mean_original_edge_rates = np.mean(original_edge_rates, axis=0)  # shape (1)
    mean_original_near_rates = np.mean(original_near_rates, axis=0)  # shape (4, 1, 1)
    mean_final_edge_rates = np.mean(final_edge_rates, axis=0)        # shape (1)
    mean_final_near_rates = np.mean(final_near_rates, axis=0)        # shape (4, 1, 1)

    # To get rid of extra dimensions if necessary, you can squeeze:
    mean_original_near_rates = np.squeeze(mean_original_near_rates, axis=(1, 2))  # shape (4,)
    mean_final_near_rates = np.squeeze(mean_final_near_rates, axis=(1, 2))        # shape (4,)

    # Average the initial outage rates across all realizations (pre-optimization)
    mean_outage_rates_near_initial = np.mean(outage_rates_near_all_realizations_initial, axis=0)  # Keep shape (4,)
    mean_outage_rates_edge_initial = np.mean(outage_rates_edge_all_realizations_initial)

    # Average the final outage rates across all realizations (post-optimization)
    mean_outage_rates_near_final = np.mean(outage_rates_near_all_realizations_final, axis=0)  # Keep shape (4,)
    mean_outage_rates_edge_final = np.mean(outage_rates_edge_all_realizations_final)

    # Print the calculated mean rates for each of the 4 base stations
    print("Mean Original Edge Rates (for each base station):", mean_original_edge_rates)
    print("Mean Final Edge Rates (for each base station):", mean_final_edge_rates)
    print("Mean Original Near Rates (for each base station):", mean_original_near_rates)
    print("Mean Final Near Rates (for each base station):", mean_final_near_rates)
        
    # Print the initial outage rates
    print("Mean Outage Rates Near Initial:", mean_outage_rates_near_initial)
    print("Mean Outage Rates Edge Initial:", mean_outage_rates_edge_initial)

    # Print the final outage rates
    print("Mean Outage Rates Near Final:", mean_outage_rates_near_final)
    print("Mean Outage Rates Edge Final:", mean_outage_rates_edge_final)
    # Calculate energy efficiency based on the averaged initial outage rates (pre-optimization)
    mean_original_efficiency = env.calculate_energy_efficiency(mean_original_near_rates, 
                                                               mean_original_edge_rates,
                                                               mean_outage_rates_near_initial, 
                                                               mean_outage_rates_edge_initial)

    # Calculate energy efficiency based on the averaged final outage rates (post-optimization)
    mean_final_efficiency = env.calculate_energy_efficiency(mean_final_near_rates, 
                                                            mean_final_edge_rates,
                                                            mean_outage_rates_near_final, 
                                                            mean_outage_rates_edge_final)
    # Print the calculated mean rates
    # Print energy efficiency before and after optimization
    print("Mean Energy Efficiency Before Optimization:", mean_original_efficiency)
    print("Mean Energy Efficiency After Optimization:", mean_final_efficiency)
# Function to calculate the average number of indices in each sublist
    def calculate_average_indices(comp_index_list):
        counts = [len(sublist) for sublist in comp_index_list]
        average_count = sum(counts) / len(counts)
        return average_count

    # Calculate the average number of indices for both lists
    average_before = calculate_average_indices(comp_index_list_before)
    average_after = calculate_average_indices(comp_index_list_after)

    # Print the lists and their average number of indices
    print(f"CoMP List Before {comp_index_list_before}")
    print(f"CoMP List After {comp_index_list_after}")

    print("Average number of indices in each sublist before:", average_before)
    print("Average number of indices in each sublist after:", average_after)

    print("Evaluation complete for single checkpoint.")

# Call the function to evaluate the single checkpoint

evaluate_single_checkpoint(algorithm, test_env_instance, best_checkpoint_path)


# Call the function to evaluate the single checkpoint
