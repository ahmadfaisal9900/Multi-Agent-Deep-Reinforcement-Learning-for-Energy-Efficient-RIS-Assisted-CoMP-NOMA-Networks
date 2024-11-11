import numpy as np
from comyx.utils import generate_seed, dbm2pow, db2pow
from comyx.network import BaseStation, UserEquipment, RIS, Link
from comyx.propagation import get_noise_power
from comyx.network import effective_channel_gain
from itertools import combinations, product
from tqdm import tqdm
import time
import pandas as pd

# Environment setup
bandwidth = 1e6
frequency = 2.4e9
temperature = 300
mc = 10
K = 3

Pt_dbm = 18
Pt_lin = dbm2pow(Pt_dbm)

N0 = get_noise_power(temperature, bandwidth)
N0_lin = dbm2pow(N0)

n_antennas = 1
n_elements = 100

nlos_fading_args = {"type": "nakagami", "m": 1, "omega": 1}
los_fading_args = {"type": "nakagami", "m": 2, "omega": 1}

pathloss_args = {
    "bu": {"type": "reference", "alpha": 3, "p0": 30, "frequency": frequency},
    "br": {"type": "reference", "alpha": 2.7, "p0": 30, "frequency": frequency},
    "ru": {"type": "reference", "alpha": 2.7, "p0": 30, "frequency": frequency},
}

# Base station, user equipment, and RIS setup
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

# Link setup
links = []
for i, bs in enumerate([BS1, BS2, BS3, BS4]):
    uc = [UC1, UC2, UC3, UC4][i]
    ris = [R1, R2, R3, R4][i]

    link_bs_uc = Link(bs, uc, nlos_fading_args, pathloss_args["bu"], shape=(1, 1, mc), seed=generate_seed(f"{bs.id}-{uc.id}"))
    link_bs_ris = Link(bs, ris, los_fading_args, pathloss_args["br"], shape=(n_elements, 1, mc), seed=generate_seed(f"{bs.id}-{ris.id}"), rician_args={"K": db2pow(K), "order": "post"})
    link_ris_uc = Link(ris, uc, los_fading_args, pathloss_args["ru"], shape=(n_elements, 1, mc), seed=generate_seed(f"{ris.id}-{uc.id}"), rician_args={"K": db2pow(K), "order": "pre"})
    link_bs_ue = Link(bs, UE, nlos_fading_args, pathloss_args["bu"], shape=(1, 1, mc), seed=generate_seed(f"{bs.id}-{UE.id}"))
    link_ris_ue = Link(ris, UE, los_fading_args, pathloss_args["ru"], shape=(n_elements, 1, mc), seed=generate_seed(f"{ris.id}-{UE.id}"), rician_args={"K": db2pow(K), "order": "pre"})

    links.append((link_bs_uc, link_bs_ris, link_ris_uc, link_bs_ue, link_ris_ue))

# Initialize RIS phase shifts and amplitudes
shape_ris = (n_elements, mc)
for ris in [R1, R2, R3, R4]:
    ris.phase_shifts = np.random.uniform(0, 2 * np.pi, shape_ris)
    ris.amplitudes = np.ones(shape_ris)

RIS_list = [R1, R2, R3, R4]
BS_list = [BS1, BS2, BS3, BS4]
from numba import njit, prange

import numpy as np

# Numba optimized SINR calculation function
def calculate_sinr_allocation_multiple(P_t, gains_near, gains_edge, N0_lin, allocation_UEf, allocation_UEn, comp_indices):
    # Reshape gains from (4, 1, 1, 10) to (4, 10)
    gains_near_sq = np.abs(gains_near.reshape(4, 1)) ** 2  # Shape: (4, 10)
    gains_edge_sq = np.abs(gains_edge.reshape(4, 1)) ** 2  # Shape: (4, 10)

    # Reshape allocations to (4, 1) to allow broadcasting across the 10 realizations
    allocation_UEf = allocation_UEf[:, np.newaxis]  # Shape: (4, 1)
    allocation_UEn = allocation_UEn[:, np.newaxis]  # Shape: (4, 1)

    # Calculate SINR for near users (vectorized)
    sinr_near_pre = (allocation_UEf * P_t * gains_near_sq) / (allocation_UEn * P_t * gains_near_sq + N0_lin)
    sinr_near = (allocation_UEn * P_t * gains_near_sq) / N0_lin

    # Calculate rates for near users (vectorized)
    rates_near = np.log2(1 + np.mean(sinr_near, axis=1))  # Averaging SINR over realizations
    rates_near_pre = np.log2(1 + np.mean(sinr_near_pre, axis=1))

    # Vectorized signal power, intra-cell, and inter-cell interference calculation for edge users
    comp_mask = np.isin(np.arange(gains_edge_sq.shape[0]), comp_indices)  # Shape: (4,)

    # Calculate signal power and interference for edge users
    signal_power = np.sum(allocation_UEf[comp_mask] * P_t * gains_edge_sq[comp_mask], axis=0)
    intra_cell_interference = np.sum(allocation_UEn[comp_mask] * P_t * gains_edge_sq[comp_mask], axis=0)
    inter_cell_interference = np.sum(P_t * gains_edge_sq[~comp_mask], axis=0)

    # Calculate SINR and rates for edge users (vectorized)
    sinr_edge = signal_power / (intra_cell_interference + inter_cell_interference + N0_lin)
    rate_edge = np.log2(1 + np.mean(sinr_edge))  # Average SINR over realizations

    return rate_edge, rates_near, rates_near_pre

"""
def recalculate_gains(links, RIS_list, realization_idx):
    gains_near = []
    gains_edge = []

    for i in range(len(RIS_list)):
        current_amplitudes = RIS_list[i].amplitudes[:, realization_idx]
        phase_shifts = RIS_list[i].phase_shifts

        reflection_matrix = np.diag(current_amplitudes * np.exp(1j * phase_shifts))

        link_bs_uc, link_bs_ris, link_ris_uc, link_bs_ue, link_ris_ue = links[i]

        gain_eff_bs_uc = link_bs_uc.channel_gain[..., realization_idx] + (link_ris_uc.channel_gain[..., realization_idx].T @ reflection_matrix @ link_bs_ris.channel_gain[..., realization_idx])
        gain_eff_bs_ue = link_bs_ue.channel_gain[..., realization_idx] + (link_ris_ue.channel_gain[..., realization_idx].T @ reflection_matrix @ link_bs_ris.channel_gain[..., realization_idx])

        gains_near.append(gain_eff_bs_uc)
        gains_edge.append(gain_eff_bs_ue)

    gains_near = np.array(gains_near)
    gains_edge = np.array(gains_edge)

    return gains_near, gains_edge
"""

def recalculate_gains(links, RIS_list, realization_idx):
    gains_near = []
    gains_edge = []

    for i in range(len(RIS_list)):
        current_amplitudes = RIS_list[i].amplitudes[:, realization_idx]
        phase_shifts = RIS_list[i].phase_shifts

        # Ensure phase_shifts is 1D and adjust its shape
        if phase_shifts.ndim == 2:
            phase_shifts = phase_shifts[:, realization_idx]

        # Create the reflection matrix
        reflection_matrix = np.diag(current_amplitudes * np.exp(1j * phase_shifts))

        link_bs_uc, link_bs_ris, link_ris_uc, link_bs_ue, link_ris_ue = links[i]

        # Extract channel gains for the given realization index
        bs_uc_gain = link_bs_uc.channel_gain[..., realization_idx].flatten()  # Flatten to (100,)
        ris_uc_gain = link_ris_uc.channel_gain[..., realization_idx]  # Shape (100, 1)
        bs_ris_gain = link_bs_ris.channel_gain[..., realization_idx]  # Shape (100, 1)

        # Ensure ris_uc_gain and bs_ris_gain are compatible for matrix multiplication
        ris_uc_gain = ris_uc_gain.flatten()  # Shape (100,)
        bs_ris_gain = bs_ris_gain.flatten()  # Shape (100,)

        # Calculate effective gains
        gain_eff_bs_uc = bs_uc_gain + (ris_uc_gain.T @ reflection_matrix @ bs_ris_gain)
        gain_eff_bs_ue = link_bs_ue.channel_gain[..., realization_idx].flatten() + \
            (link_ris_ue.channel_gain[..., realization_idx].flatten().T @ reflection_matrix @ bs_ris_gain)

        gains_near.append(gain_eff_bs_uc)
        gains_edge.append(gain_eff_bs_ue)

    gains_near = np.array(gains_near)
    gains_edge = np.array(gains_edge)

    return gains_near, gains_edge


def calculate_rates(P_t, RIS_list, links, N0_lin, allocation, comp_indices, realization_idx):
    # Call recalculate_gains to get the gains for near and edge users
    gains_near, gains_edge = recalculate_gains(links, RIS_list, realization_idx)

    # Calculate SINR and rates using the gains from recalculate_gains
    rate_edge, rates_near, rates_near_pre = calculate_sinr_allocation_multiple(
        P_t, gains_near, gains_edge, N0_lin,
        np.array(allocation["UEf"]), np.array(allocation["UEn"]),
        comp_indices
    )

    return rate_edge, rates_near, rates_near_pre

def calculate_outage_probability(rate, target_rate):
    # Vectorized calculation for outage probability
    target_snr = 2 ** target_rate - 1
    actual_snr = 2 ** rate - 1
    return (actual_snr < target_snr).astype(int)  # Vectorized comparison


def calculate_energy_efficiency(P_t, rates_near, rate_far, near_outage_prob, far_outage_prob, P_Q, P_R, lambda_pa, comp_indices, BS_list):
        # Effective outage rates for near users
        effective_outage_rates_near = [(1 - near_outage_prob[i]) * rates_near[i] for i in range(len(rates_near))]
        total_effective_rate_near = sum(effective_outage_rates_near)

        # Effective outage rate for far user
        effective_outage_rate_far = (1 - far_outage_prob) * rate_far  # Only one far user

        # Total power consumption for near users (all base stations excluding CoMP)
        power_near = len(BS_list) * ((1 / lambda_pa) * Pt_lin + P_Q)

        # Total power consumption for CoMP base stations (serving far user)
        power_comp = len(comp_indices) * ((1 / lambda_pa) * Pt_lin + P_Q + P_R)

        # Energy efficiency calculation
        energy_efficiency_near = total_effective_rate_near / power_near
        energy_efficiency_far = effective_outage_rate_far / power_comp

        # Total energy efficiency
        total_energy_efficiency = energy_efficiency_near + energy_efficiency_far

        return total_energy_efficiency


def optimize_phase_shifts_inner_mc(Pt_lin, links, RIS_list, N0_lin, allocation, comp_indices, quantized_phase_shifts, mc, min_rate_threshold_edge=1.2, min_rate_threshold_near=1.5):
    n_RIS = len(RIS_list)
    n_elements = RIS_list[0].phase_shifts.shape[0]

    # Initialize best combination and rates
    best_combination = [[None] * n_elements for _ in range(n_RIS)]

    # Best average rates initialization
    best_avg_rate_edge = -np.inf
    best_avg_rates_near = -np.inf * np.ones(len(links))

    # Iterate over each element in each RIS
    for element_idx in tqdm(range(n_elements), desc="Optimizing Elements"):
        for ris_idx in range(n_RIS):
            # Local best initialization
            best_local_rate_edge = -np.inf
            best_local_rates_near = -np.inf * np.ones(len(links))
            current_RIS = RIS_list[ris_idx]

            # Accumulate rates for averaging across MC realizations
            accumulated_rate_edge = np.zeros(mc)
            accumulated_rates_near = np.zeros((len(links), mc))

            # Iterate over each phase
            for phase in quantized_phase_shifts:
                # Assign phase shift to the current element of the current RIS
                current_RIS.phase_shifts[element_idx] = phase

                # Debug statement for current phase assignment
                #print(f"Evaluating Phase {phase} for RIS {ris_idx}, Element {element_idx}")

                # Iterate over each Monte Carlo (MC) realization for this phase
                for realization_idx in range(mc):
                    # Calculate rates based on the current phase shift configuration
                    rate_edge, rates_near, rates_near_pre = calculate_rates(
                        Pt_lin, RIS_list, links, N0_lin, allocation, comp_indices, realization_idx
                    )

                    # Accumulate rates for averaging
                    accumulated_rate_edge[realization_idx] = rate_edge
                    accumulated_rates_near[:, realization_idx] = rates_near

                # After evaluating all MC realizations for this phase, compute average rates
                avg_rate_edge = np.mean(accumulated_rate_edge)
                avg_rates_near = np.mean(accumulated_rates_near, axis=1)

                # Debug statement for average rates after MC realizations
                #print(f"Avg Edge Rate for Phase {phase}: {avg_rate_edge}, Avg Near Rates: {avg_rates_near}")

                # Ensure both average rates exceed minimum thresholds
                if avg_rate_edge > min_rate_threshold_edge and avg_rates_near[ris_idx] > min_rate_threshold_near:
                    # Check if the new configuration improves the rates
                    #print(f"Checking if the new configuration improves: Local Edge Rate: {best_local_rate_edge}, New Edge Rate: {avg_rate_edge}")
                    #print(f"Checking if the new configuration improves: Local Near Rates: {best_local_rates_near}, New Near Rates: {avg_rates_near}")

                    if avg_rate_edge > best_local_rate_edge and avg_rates_near[ris_idx] > best_local_rates_near[ris_idx]:

                        best_local_rate_edge = avg_rate_edge
                        best_local_rates_near = avg_rates_near.copy()

                        # Store the best phase shift for this element of the RIS
                        best_combination[ris_idx][element_idx] = phase

                        # Debug statement for updated best configuration
                        #print(f"Updated Best Configuration for RIS {ris_idx}, Element {element_idx}: Phase {phase}")

    # Convert lists of phase shifts to 1D numpy arrays for each RIS if needed
    best_combination = [np.array(phases) for phases in best_combination]

    # Print the best phase shift combinations and corresponding average rates
    print("Best Phase Shift Combinations:", best_combination)
    print(f"    Best Avg Edge Rate: {best_local_rate_edge}")
    print(f"    Best Avg Near Rates: {best_local_rates_near}")

    # Return the best phase configuration and the corresponding average rates
    return best_combination, best_avg_rate_edge, best_avg_rates_near

def optimize_power_allocation(Pt_lin, RIS_list, links, N0_lin, P_Q, P_R, lambda_pa, BS_list, best_combination, mc, quantized_phase_shifts, min_rate_threshold_edge=1.2, min_rate_threshold_near=1.5):
    power_allocation_steps = np.arange(0.5, 1.0, 0.05)
    best_energy_efficiency = -np.inf
    best_power_allocation = None
    best_comp_indices = None
    best_rate_edge = None
    best_rates_near = None
    max_edge_rate = -np.inf  # Initialize to track the maximum edge rate
    max_rates_near = None  # To track corresponding near rates for max edge rate
    max_energy_efficiency = -np.inf  # Track max energy efficiency for max edge rate
    max_power_allocation = None  # Track best power allocation for max edge rate
    max_comp_combination = None  # Track best CoMP combination for max edge rate
    best_total_score = -np.inf
    r_power_allocation = None
    r_comp_indices = None
    r_rate_edge = None
    r_rates_near = None
    r_best_rate = None


    # Constants for weights
    w_near = 1  # Fixed weight for near user rates
    w_edge = 32  # Fixed weight for edge user rates

    # Apply the optimized phase shifts to all RISes
    for ris_idx, ris in enumerate(RIS_list):
        ris.phase_shifts = best_combination[ris_idx]

    # Iterate over all CoMP combinations with a progress bar
    for comp_comb in tqdm(comp_combinations, desc="Optimizing Power Allocation and CoMP"):
        # Use tqdm for power allocation combinations
        for allocation_combination in tqdm(product(power_allocation_steps, repeat=len(links)), desc="Processing Allocations", leave=False):
            allocation = {
                "UEn": [1 - UEf for UEf in allocation_combination],  # Near user allocation
                "UEf": list(allocation_combination)  # Far user allocation
            }

            # Initialize lists to collect rates across realizations
            edge_rates = []
            near_rates = []
            edge_outage_probs = []
            near_outage_probs = []

            for realization_idx in range(mc):  # Removed tqdm from this loop
                rate_edge, rates_near, _ = calculate_rates(
                    Pt_lin, RIS_list, links, N0_lin, allocation, comp_comb, realization_idx
                )

                # Store rates for averaging later
                edge_rates.append(rate_edge)
                near_rates.append(rates_near)

                # Calculate outage probability for this realization
                outage_near = calculate_outage_probability(rates_near, min_rate_threshold_near)
                outage_edge = calculate_outage_probability(rate_edge, min_rate_threshold_edge)

                # Append outage probabilities for later summation
                edge_outage_probs.append(outage_edge)
                near_outage_probs.append(outage_near)

            # Compute average rates across all realizations
            avg_rate_edge = np.mean(edge_rates)
            avg_rates_near = np.mean(near_rates, axis=0)
            sum_avg_near_rates = np.sum(avg_rates_near)

            # Compute the weighted total score
            total_score = sum_avg_near_rates + avg_rate_edge

            # Sum and average the outage probabilities across all realizations
            avg_outage_prob_edge = np.mean(edge_outage_probs)
            avg_outage_prob_near = np.mean(near_outage_probs, axis=0)

            # Calculate energy efficiency
            energy_efficiency = calculate_energy_efficiency(
                Pt_lin, avg_rates_near, avg_rate_edge,
                avg_outage_prob_near,
                avg_outage_prob_edge,
                P_Q, P_R, lambda_pa, comp_comb, BS_list
            )

            # Update if energy efficiency improves
            if (avg_rates_near > min_rate_threshold_near).all() and (avg_rate_edge > min_rate_threshold_edge):
                if energy_efficiency > best_energy_efficiency:
                    best_energy_efficiency = energy_efficiency
                    best_power_allocation = allocation
                    best_comp_indices = comp_comb
                    best_rate_edge = avg_rate_edge
                    best_rates_near = avg_rates_near

                # Check if this is the maximum edge rate
                if avg_rate_edge > max_edge_rate:
                    max_edge_rate = avg_rate_edge
                    max_rates_near = avg_rates_near
                    max_energy_efficiency = energy_efficiency
                    max_power_allocation = allocation
                    max_comp_combination = comp_comb

                # Check if this configuration improves the total score
                if total_score > best_total_score:
                    best_total_score = total_score
                    r_power_allocation = allocation
                    r_comp_indices = comp_comb
                    r_rate_edge = avg_rate_edge
                    r_rates_near = avg_rates_near
                    r_best_rate = np.sum(r_rates_near) + r_rate_edge


    print(f"Updated Best Energy Efficiency: {best_energy_efficiency}")
    print(f"Best Power Allocation: {best_power_allocation}")
    print(f"Best CoMP Combination: {best_comp_indices}")
    print(f"Best Edge Rate: {best_rate_edge}")
    print(f"Best Near Rates: {best_rates_near}")
    print()

    print(f"Updated Maximum Edge Rate: {max_edge_rate}")
    print(f"Corresponding Near Rates: {max_rates_near}")
    print(f"Corresponding Energy Efficiency: {max_energy_efficiency}")
    print(f"Corresponding Power Allocation: {max_power_allocation}")
    print(f"Corresponding CoMP Combination: {max_comp_combination}")
    print()

    print(f"Updated Best Total Score: {r_best_rate}")
    print(f"Best Power Allocation: {r_power_allocation}")
    print(f"Best CoMP Combination: {r_comp_indices}")
    print(f"Best Edge Rate: {r_rate_edge}")
    print(f"Best Near Rates: {r_rates_near}")
    print()


    # Return the best power allocation, CoMP combination, energy efficiency, and the max edge rate and its corresponding values
    return best_power_allocation, best_comp_indices, best_energy_efficiency, best_rate_edge, best_rates_near, max_edge_rate, max_rates_near, max_energy_efficiency, max_power_allocation, max_comp_combination, r_power_allocation, r_comp_indices, r_rate_edge, r_rates_near, r_best_rate

# Function to save results to CSV
def save_results_to_csv(filename, phase_shifts_results, power_allocation_results):
    # Organize the data into a dictionary for easier CSV export using pandas
    data = {
        'Best Phase Combination': [phase_shifts_results[0]],

        'Best Energy Efficiency (1)': [power_allocation_results[2]],
        'Corresponding Power Allocation (1)': [power_allocation_results[0]],
        'Corresponding CoMP Indices (1)': [power_allocation_results[1]],
        'Corresponding Edge Rate (1)': [power_allocation_results[3]],
        'Corresponding Rates (1)': [power_allocation_results[4]],

        'Best Edge Rate (2)': [power_allocation_results[5]],
        'Corresponding Rates Near (2)': [power_allocation_results[6]],
        'Corresponding Energy Efficiency (2)': [power_allocation_results[7]],
        'Corresponding Power Allocation (2)': [power_allocation_results[8]],
        'Corresponding CoMP Combination (2)': [power_allocation_results[9]],

        'Best Total Rate (3)': [power_allocation_results[14]],
        'Corresponding Power Allocation (3)': [power_allocation_results[10]],
        'Corresponding CoMP Indices (3)': [power_allocation_results[11]],
        'Corresponding Edge Rate (3)': [power_allocation_results[12]],
        'Corresponding Near Rates (3)': [power_allocation_results[13]],
    }


    # Create a pandas DataFrame
    df = pd.DataFrame(data)

    # Write the DataFrame to a CSV file
    df.to_csv(filename, index=False)

# Initialize parameters
quantized_phase_shifts = np.linspace(0, 2 * np.pi, 36, endpoint=False)
allocation = {"UEn": [0.25, 0.15, 0.2, 0.15], "UEf": [0.75, 0.85, 0.8, 0.85]}
Pt_dbm = 18
Pt_lin = dbm2pow(Pt_dbm)
comp_indices = [0, 1, 2, 3]  # Example CoMP configuration
P_Q = 30  # Example base station power consumption
PQ_lin = dbm2pow(P_Q)
P_R = 10  # Example RIS power consumption
PR_lin = dbm2pow(P_R)
lambda_pa = 0.9
mc = 10
# Function to save results to a CSV file
import csv

def save_results_to_csv(filename, results):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Transmit Power (dBm)", "Best Combination", "Best Avg Rate Edge", "Best Avg Rates Near",
                         "Best Power Allocation", "Best CoMP Indices", "Best Energy Efficiency", "Best Rate Edge",
                         "Best Rates Near", "Max Edge Rate", "Max Rates Near", "Max Energy Efficiency",
                         "Max Power Allocation", "Max CoMP Combination", "R Power Allocation", "R CoMP Indices",
                         "R Rate Edge", "R Rates Near", "R Best Rate"])
        for row in results:
            writer.writerow(row)
# Number of base stations and CoMP combinations
num_base_stations = 4
max_comp_base_stations = 5
comp_combinations = []
for r in range(1, max_comp_base_stations + 1):
    comp_combinations.extend(combinations(range(num_base_stations), r))

# Store results
results = []

# Loop through Pt_dbm from 0 to 20
for Pt_dbm in range(0, 21, 2):  # 0 to 20 dBm, incrementing by 2
    Pt_lin = dbm2pow(Pt_dbm)  # Convert to linear scale

    # Call the optimize_phase_shifts function with mc = 10 realizations
    best_combination, best_avg_rate_edge, best_avg_rates_near = optimize_phase_shifts_inner_mc(
        Pt_lin, links, RIS_list, N0_lin, allocation, comp_indices, quantized_phase_shifts, mc,
        min_rate_threshold_edge=1.2, min_rate_threshold_near=1.5
    )

    # Call the optimize_power_allocation function
    best_power_allocation, best_comp_indices, best_energy_efficiency, best_rate_edge, best_rates_near, \
    max_edge_rate, max_rates_near, max_energy_efficiency, max_power_allocation, max_comp_combination, \
    r_power_allocation, r_comp_indices, r_rate_edge, r_rates_near, r_best_rate = optimize_power_allocation(
        Pt_lin, RIS_list, links, N0_lin, PQ_lin, PR_lin, lambda_pa, BS_list, best_combination, mc,
        quantized_phase_shifts, min_rate_threshold_edge=1.2, min_rate_threshold_near=1.5
    )

    # Append results for the current Pt_dbm
    results.append([
        Pt_dbm, best_combination, best_avg_rate_edge, best_avg_rates_near,
        best_power_allocation, best_comp_indices, best_energy_efficiency, best_rate_edge, best_rates_near,
        max_edge_rate, max_rates_near, max_energy_efficiency, max_power_allocation, max_comp_combination,
        r_power_allocation, r_comp_indices, r_rate_edge, r_rates_near, r_best_rate
    ])

# Save the results to CSV
save_results_to_csv('exhaustiveSearch_results_0to20dbm.csv', results)