import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gymnasium import spaces
from comyx.network import BaseStation, UserEquipment, RIS, Link
from comyx.propagation import get_noise_power
from comyx.utils import dbm2pow, generate_seed, db2pow
import gc
import torch

class MultiAgentBaseStationEnv(MultiAgentEnv):
    def __init__(self, env_config):
        self.BS_list = env_config["BS_list"]
        self.RIS_list = env_config["RIS_list"]
        self.UC_list = env_config["UC_list"]
        self.UE = env_config["UE"]
        self.links = env_config["links"]
        self.mc = env_config["mc"]
        self.comp_indices = env_config["comp_indices"]
        self.og_comp_indices = self.comp_indices
        self.threshold_edge = env_config["threshold_edge"]
        self.threshold_near = 1.5
        self.max_steps = 200
        self.n_elements = 100  # Assuming 100 elements for the RIS
        self.shape_ris = (self.n_elements,)  # Correct shape for phase shifts

        self.Pt = 10 # Fixed power level in dBm
        self.Pt_lin = dbm2pow(self.Pt)  # Convert to Watt
        self.N0 = get_noise_power(300, 1e6)  # dBm
        self.N0_lin = dbm2pow(self.N0)  # Watt
        self.efficiency_threshold = 1000000

        self.lambda_pa = 0.4  # Power amplifier efficiency
        self.P_Q = dbm2pow(30)  # Operating power of the BS in Watts
        self.Pele = dbm2pow(5)  # Power consumption of each RIS element in Watts
        self.P_R = self.n_elements * self.Pele  # Total power consumption of RIS in Watts
        self.testing_mode = False
        self.action_space = {
            f"BS{i}": spaces.Box(
                low=np.concatenate((-np.pi * np.ones(self.shape_ris), [0.5])),
                high=np.concatenate((np.pi * np.ones(self.shape_ris), [1.0])),
                dtype=np.float64
            ) for i in range(1, len(self.BS_list) + 1)
        }

        # Define discrete action space for CoMP
        self.action_space["CoMP"] = spaces.Discrete(2 ** len(self.BS_list))

        num_gains = 2  # Normalized gains for optimized BS to its UC and the edge UE
        num_power_allocations = 1  # Only one power level optimized
        # Updated observation space to include outage rates
        self.observation_space = {
            f"BS{i}": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(7 + 100,),  # 107 elements (additional element for rate)
                dtype=np.float64
            ) for i in range(1, len(self.BS_list) + 1)
        }
        # CoMP agent needs to consider observations of all BS plus edge user and their outage probabilities and energy efficiency
        self.observation_space["CoMP"] = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(2 * len(self.BS_list) + 3 + 4,),  # (number of BS * rate) + rate_edge + outage_edge + energy efficiency
            dtype=np.float64
        )
        x = 2 * len(self.BS_list) + 3 + 4
        #print(f"Observation Space CoMP at the very Start {x}")
        self.fixed_power_allocations = {
            "UEn": [0.2, 0.2, 0.2, 0.2],
            "UEf": [0.8, 0.8, 0.8, 0.8]
        }
        self.energy_efficiency = 0.0
        self.agents = [f"BS{i}" for i in range(1, len(self.BS_list) + 1)] + ["CoMP"]
        self.reset()

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.realization_idx = np.random.choice(self.mc)
        self.performance_counter = 0
        self.energy_efficiency = 0.0
        self.comp_indices = self.og_comp_indices
        observations = {}
        infos = {}

        for i in range(len(self.RIS_list)):
            self.RIS_list[i].phase_shifts = np.random.uniform(0, 2 * np.pi, self.shape_ris)

        self.recalculate_gains(self.realization_idx)
        self.fixed_power_allocations = {
            "UEn": [0.2, 0.2, 0.2, 0.2],
            "UEf": [0.8, 0.8, 0.8, 0.8]
        }
        rate_edge, rates_near, rates_near_pre = self.calculate_sinr_allocation_multiple(
            self.Pt_lin,
            self.gains_near,
            self.gains_edge,
            self.N0_lin,
            self.fixed_power_allocations,
            self.comp_indices
        )
        self.update_outage_rates(rates_near, rates_near_pre, rate_edge)
        self.energy_efficiency = self.calculate_energy_efficiency(rates_near, rate_edge, self.outage_rates_near, self.outage_rate_edge)

        for agent in self.agents:
            observations[agent] = self.observe(agent, self.fixed_power_allocations)
            if np.any(np.isnan(observations[agent])) or np.any(np.isinf(observations[agent])):
                print(f"Invalid values in observation for {agent}: {observations[agent]}")

            infos[agent] = {}  # Initialize infos for each agent

        return observations, infos
    def normalize(self, value, min_val, max_val):
            """Normalize a value to the range [0, 1]."""
            return np.clip((value - min_val) / (max_val - min_val), 0, 1)
    def observe(self, agent, allocations):
        rate_edge, rates_near, rates_near_pre = self.calculate_sinr_allocation_multiple(
            self.Pt_lin,
            self.gains_near,
            self.gains_edge,
            self.N0_lin,
            allocations,
            self.comp_indices
        )
        comp_indices = self.get_comp_obs(self.comp_indices, 4)
        comp_indices = np.array(comp_indices).flatten()
        #print(f"Length of CoMP Indicies {len(comp_indices)}")
        # Flatten rates
        rates_near_flat = rates_near.flatten()
        rate_edge_flat = np.array([rate_edge]).flatten()

        # print(f"Rates Near Shape: {rates_near.shape}, Flattened: {rates_near_flat.shape}")
        # print(f"Rate Edge Shape: {np.array([rate_edge]).shape}, Flattened: {rate_edge_flat.shape}")

        if agent == "CoMP":
            outage_rates_near = np.array([self.calculate_outage_probability(r, self.threshold_edge) for r in rates_near_pre])
            outage_rate_edge = self.calculate_outage_probability(rate_edge, self.threshold_edge)

            # Flatten arrays
            outage_rates_near_flat = outage_rates_near.flatten()
            outage_rate_edge_flat = np.array([outage_rate_edge]).flatten()
            energy_efficiency_flat = np.array([self.energy_efficiency]).flatten()

            # print(f"Outage Rates Near Shape: {outage_rates_near.shape}, Flattened: {outage_rates_near_flat.shape}")
            # print(f"Outage Rate Edge Shape: {np.array([outage_rate_edge]).shape}, Flattened: {outage_rate_edge_flat.shape}")
            # print(f"Energy Efficiency Shape: {np.array([self.energy_efficiency]).shape}, Flattened: {energy_efficiency_flat.shape}")

            # Concatenate and return observation for CoMP agent
            observation = np.concatenate((
                rates_near_flat, 
                rate_edge_flat, 
                outage_rates_near_flat, 
                outage_rate_edge_flat, 
                energy_efficiency_flat,
                comp_indices
            )).astype(np.float64)

            #print(f"CoMP Observation Shape: {observation.shape}")
            return observation

        else:
            agent_index = int(agent[2:]) - 1
            # Compute individual observation components
            obs_gains_near = np.array([np.log(np.linalg.norm(self.gains_near[agent_index]))])
            obs_gains_edge = np.array([np.log(np.linalg.norm(self.gains_edge[agent_index]))])
            phase_shift_flat = self.RIS_list[agent_index].phase_shifts.flatten()
            current_power_allocation = np.array([allocations["UEf"][agent_index]])
            outage_prob_near = np.array([self.calculate_outage_probability(rates_near_pre[agent_index], self.threshold_edge)])
            outage_prob_edge = np.array([self.calculate_outage_probability(rate_edge, self.threshold_edge)])
            rate_near = np.array([rates_near[agent_index]])

            # print(f"Obs Gains Near Shape: {obs_gains_near.shape}")
            # print(f"Obs Gains Edge Shape: {obs_gains_edge.shape}")
            # print(f"Phase Shifts Shape: {self.RIS_list[agent_index].phase_shifts.shape}, Flattened: {phase_shift_flat.shape}")
            # print(f"Power Allocation Shape: {current_power_allocation.shape}")
            # print(f"Outage Prob Near Shape: {outage_prob_near.shape}")
            # print(f"Outage Prob Edge Shape: {outage_prob_edge.shape}")
            # print(f"Rate Near Shape: {rate_near.shape}, Flattened: {rate_near.flatten().shape}")
            # print(f"Rate Edge Flat Shape: {rate_edge_flat.shape}")

            # Concatenate and return observation for non-CoMP agent
            observation = np.concatenate((
                obs_gains_near, 
                obs_gains_edge, 
                phase_shift_flat, 
                current_power_allocation, 
                outage_prob_near, 
                outage_prob_edge, 
                rate_near.flatten(), 
                rate_edge_flat.flatten(), 
            )).astype(np.float64)

            #print(f"Non-CoMP Observation Shape: {observation.shape}")
            return observation


    def step(self, action_dict):
        self.current_step += 1
        obs, rewards, terminated, truncated, infos = {}, {}, {}, {}, {}
        allocation = {
            "UEn": self.fixed_power_allocations["UEn"][:],
            "UEf": self.fixed_power_allocations["UEf"][:]
        }
        # Handle discrete actions for CoMP
        comp_action = action_dict.pop("CoMP")
        comp_decision = [int(x) for x in bin(comp_action)[2:].zfill(len(self.BS_list))]
        
        # Enforce at least one base station must participate in CoMP
        if np.sum(comp_decision) == 0:
            comp_decision[np.random.choice(len(comp_decision))] = 1

        self.comp_indices = [i for i, x in enumerate(comp_decision) if x == 1]
        for agent, action in action_dict.items():
            agent_index = int(agent[2:]) - 1
            phase_shift_action = action[:-1]
            current_power_allocation = action[-1]

            self.RIS_list[agent_index].phase_shifts = phase_shift_action

            self.recalculate_gains(self.realization_idx)

            self.RIS_list[agent_index].phase_shifts = phase_shift_action
            allocation["UEn"][agent_index] = 1 - current_power_allocation
            allocation["UEf"][agent_index] = current_power_allocation
            self.fixed_power_allocations = allocation

        rate_edge, rates_near, rates_near_pre = self.calculate_sinr_allocation_multiple(
            self.Pt_lin,
            self.gains_near,
            self.gains_edge,
            self.N0_lin,
            allocation,
            self.comp_indices
        )
        self.update_outage_rates(rates_near, rates_near_pre, rate_edge)
        self.energy_efficiency = self.calculate_energy_efficiency(rates_near, rate_edge, self.outage_rates_near, self.outage_rate_edge)
        
        #max_efficiency = 8.0  # Maximum expected value of energy efficiency for normalization
        #normalized_energy_efficiency = self.energy_efficiency / max_efficiency

        for agent in self.agents:
            if agent == "CoMP":
                rewards[agent] = self.calculate_comp_agent_reward(self.energy_efficiency, rates_near, rate_edge)
            else:
                agent_index = int(agent[2:]) - 1
                comp_reward = self.comp_reward(rate_edge, rates_near[agent_index])
                max_comp_reward = 15.0  # Maximum expected value of comp reward for normalization
                normalized_comp_reward = comp_reward / max_comp_reward
                rewards[agent] = normalized_comp_reward
            obs[agent] = self.observe(agent, allocation)
            terminated[agent] = self.current_step >= self.max_steps
            truncated[agent] = False
            infos[agent] = {}

        terminated["__all__"] = all(terminated.values())
        truncated["__all__"] = False
        #print(rewards["CoMP"])
        return obs, rewards, terminated, truncated, infos
    def calculate_comp_agent_reward(self, energy_efficiency, near_rate, edge_rate, rate_threshold=1.2, penalty_weight=100):
        # Existing reward based on energy efficiency
        reward = energy_efficiency
    
        near_rate = near_rate.flatten()
        for near_rate_ in near_rate:
            if near_rate_ < (rate_threshold + 0.3):
                reward -= penalty_weight * (rate_threshold - near_rate_)
        
        edge_rate = edge_rate.flatten()
        if np.any(edge_rate < rate_threshold):
            reward -= penalty_weight * (rate_threshold - edge_rate)
        scaling_factor = 4
        # Ensure reward doesn't go below a certain minimum, if desired
        #reward = max(reward, -1000)  # For example, prevent extreme negative rewards
        reward = float(reward)
        reward = reward/scaling_factor
        assert isinstance(reward, (int, float)), f"Reward must be int or float, got {type(reward)}"
        return reward
    
    def comp_reward(self, rate_edge, rate_near):
        # Calculate the sum of rates
        Rsum = rate_edge + rate_near

        # Calculate how close the rates are to the thresholds
        edge_closeness = rate_edge / self.threshold_edge
        near_closeness = rate_near / self.threshold_near

        # Check if both rates are above their respective thresholds
        if rate_edge >= self.threshold_edge and rate_near >= self.threshold_near:
            # Reward is simply the sum rate when both QoS conditions are met
            reward = Rsum
        else:
            # Apply a penalty if either rate is below its threshold
            max_penalty = -15
            penalty_scaling_factor = 0.5  # Adjust how steeply the penalty is applied
            continuous_penalty = max_penalty * (1 - penalty_scaling_factor * (edge_closeness + near_closeness) / 2)

            # Ensure the penalty does not exceed the maximum allowed penalty
            continuous_penalty = max(continuous_penalty, max_penalty)

            # Total reward is the sum rate with a penalty
            reward = continuous_penalty + Rsum

        return reward


    # def comp_reward(self, rate_edge, rate_near):
    #     # Normalizing rates
    #     Rsum = rate_edge + rate_near

    #     # Calculate how close the rates are to the threshold
    #     edge_closeness = rate_edge / self.threshold_edge
    #     near_closeness = rate_near / self.threshold_near

    #     # Linearly decrease the penalty as rates approach the threshold
    #     max_penalty = -15
    #     penalty_scaling_factor = 0.5  # Adjust this factor to control how quickly the penalty decreases
    #     continuous_penalty = max_penalty * (1 - penalty_scaling_factor * (edge_closeness + near_closeness) / 2)

    #     # Ensure continuous_penalty does not exceed the maximum penalty
    #     continuous_penalty = max(continuous_penalty, max_penalty)

    #     # Large reward for exceeding both thresholds
    #     large_reward = 15 if rate_edge >= self.threshold_edge and rate_near >= self.threshold_near else 0

    #     # Total reward
    #     reward = continuous_penalty + Rsum + large_reward
    #     return reward

    def recalculate_gains(self, realization_idx):
        current_amplitudes = np.array([self.RIS_list[i].amplitudes[:, realization_idx] for i in range(len(self.BS_list))])
        phase_shifts = np.array([self.RIS_list[i].phase_shifts for i in range(len(self.BS_list))])

        # Create reflection matrices for all RIS elements
        reflection_matrices = np.array([np.diag(current_amplitudes[i] * np.exp(1j * phase_shifts[i])) for i in range(len(self.BS_list))])

        # Unpack links for all base stations
        link_bs_uc, link_bs_ris, link_ris_uc, link_bs_ue, link_ris_ue = zip(*self.links)

        # Calculate effective gains for near and edge users
        gain_eff_bs_uc = np.array([
            link_bs_uc[i].channel_gain[..., realization_idx] + 
            (link_ris_uc[i].channel_gain[..., realization_idx].T @ reflection_matrices[i] @ link_bs_ris[i].channel_gain[..., realization_idx])
            for i in range(len(self.BS_list))
        ])
        
        gain_eff_bs_ue = np.array([
            link_bs_ue[i].channel_gain[..., realization_idx] + 
            (link_ris_ue[i].channel_gain[..., realization_idx].T @ reflection_matrices[i] @ link_bs_ris[i].channel_gain[..., realization_idx])
            for i in range(len(self.BS_list))
        ])

        self.gains_near = gain_eff_bs_uc
        self.gains_edge = gain_eff_bs_ue

    def calculate_sinr_allocation_multiple(self, P_t, gains_near, gains_edge, N0_lin, allocation, comp_indices):
        sinr_near = np.zeros_like(gains_near, dtype=np.float64)
        sinr_near_pre = np.zeros_like(gains_near, dtype=np.float64)
        rates_near = np.zeros_like(gains_near, dtype=np.float64)
        rates_near_pre = np.zeros_like(gains_near, dtype=np.float64)

        for i in range(len(gains_near)):
            sinr_near_pre[i] = (allocation["UEf"][i] * P_t * np.abs(gains_near[i])**2) / (
                allocation["UEn"][i] * P_t * np.abs(gains_near[i])**2 + N0_lin)
            sinr_near[i] = (allocation["UEn"][i] * P_t * np.abs(gains_near[i])**2) / N0_lin
            rates_near[i] = np.log2(1 + sinr_near[i])
            rates_near_pre[i] = np.log2(1 + sinr_near_pre[i])

        signal_power = np.zeros_like(gains_edge[0], dtype=np.float64)
        intra_cell_interference = np.zeros_like(gains_edge[0], dtype=np.float64)
        inter_cell_interference = np.zeros_like(gains_edge[0], dtype=np.float64)

        for i in range(len(gains_edge)):
            allocation_factor_near = allocation["UEn"][i]
            allocation_factor_far = allocation["UEf"][i]
            channel_power = np.abs(gains_edge[i])**2

            if i in comp_indices:
                signal_power += allocation_factor_far * P_t * channel_power
                intra_cell_interference += allocation_factor_near * P_t * channel_power
            else:
                inter_cell_interference += P_t * channel_power
        
        sinr_edge = signal_power / (intra_cell_interference + inter_cell_interference + N0_lin)
        rate_edge = np.log2(1 + sinr_edge)
        return rate_edge, rates_near, rates_near_pre

    def calculate_energy_efficiency(self, rates_near, rate_far, near_outage_prob, far_outage_prob):
        # Effective outage rates for near users
        effective_outage_rates_near = [(1 - near_outage_prob[i]) * rates_near[i] for i in range(len(rates_near))]
        total_effective_rate_near = sum(effective_outage_rates_near)
        
        # Effective outage rate for far user
        effective_outage_rate_far = (1 - far_outage_prob) * rate_far  # Only one far user
        
        # Total power consumption for near users (all base stations excluding CoMP)
        power_near = len(self.BS_list) * ((1 / self.lambda_pa) * self.Pt_lin + self.P_Q)
        
        # Total power consumption for CoMP base stations (serving far user)
        power_comp = len(self.comp_indices) * ((1 / self.lambda_pa) * self.Pt_lin + self.P_Q + self.P_R)
        
        # Energy efficiency calculation
        energy_efficiency_near = total_effective_rate_near / power_near
        energy_efficiency_far = effective_outage_rate_far / power_comp
        
        # Total energy efficiency
        total_energy_efficiency = energy_efficiency_near + energy_efficiency_far
        
        return float(total_energy_efficiency)

    def calculate_outage_probability(self, rate, target_rate):
        # Calculate the SINR threshold for the target rate
        target_snr = 2 ** target_rate - 1
        actual_snr = 2 ** rate - 1
        return 1 if actual_snr < target_snr else 0

    def update_outage_rates(self, rates_near, rates_near_pre, rate_edge):
        # Update the outage rates based on the current rates and thresholds
        self.outage_rates_near = [
            1 if self.calculate_outage_probability(rate_near, self.threshold_near) or self.calculate_outage_probability(rate_pre, self.threshold_edge)
            else 0 for rate_near, rate_pre in zip(rates_near, rates_near_pre)
        ]
        self.outage_rate_edge = self.calculate_outage_probability(rate_edge, self.threshold_edge)
    def update_channel_realization(self):
        # Update just the channel realization index and recalculate gains
        self.realization_idx = np.random.choice(self.mc)
        self.recalculate_gains(self.realization_idx)

    def render(self):
        pass

    def close(self):
        pass
    def get_comp_obs(self,comp_indices, n_bs):
        # Initialize a list of zeros of length n_bs
        comp_obs = [0] * n_bs
        
        # Set the indices in comp_indices to 1
        for idx in comp_indices:
            comp_obs[idx] = 1
        
        return comp_obs