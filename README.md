# Multi-Agent Deep Reinforcement Learning for Energy Efficient RIS-Assisted CoMP-NOMA Networks

This repository contains the code and configurations for a novel multi-agent deep reinforcement learning (MADRL) approach aimed at optimizing energy efficiency and user rates in reflective intelligent surface (RIS)-assisted coordinated multi-point (CoMP) non-orthogonal multiple access (NOMA) networks. 

## Project Structure

- `CoMP_MADRL.py`: Defines the environment class `MultiAgentBaseStationEnv` for the CoMP-NOMA network, with configurations for base stations, user equipment, RIS elements, and channel parameters.
- `CoMPTest.py`: Testing script that initializes the trained MADRL model and evaluates performance metrics like outage rates, energy efficiency, and user rates.
- `CoMP_Tune.py`: Training script that uses Ray's Tune library to configure and run the training of the MADRL model with Proximal Policy Optimization (PPO).

## Getting Started

1. **Environment Setup**:  
   The environment is configured using Ray's RLlib library and various parameters like transmit power, noise power, fading types, and path loss models. The environment allows each agent (base station) to optimize RIS phase shifts, power allocations, and CoMP participation.

2. **Training**:
   - Run `CoMP_Tune.py` to train the MADRL model. The script leverages PPO with a custom reward structure to achieve optimal rates and energy efficiency.
   - Training logs are stored for performance monitoring, and checkpoints are saved periodically.

3. **Testing**:
   - Run `CoMPTest.py` to load a trained model and perform extensive performance evaluation. The script calculates metrics such as SINR, outage rates, and energy efficiency for each agent and the entire network.

## Key Components

- **Environment** (`CoMP_MADRL.py`): 
   - `MultiAgentBaseStationEnv`: A custom environment with agents for each base station and RIS elements. Agents optimize local parameters to meet quality-of-service (QoS) and energy efficiency targets.
   - Observations include factors like RIS phase shifts, power allocations, SINR, and outage rates.
   - The action space includes power allocation factors, RIS phase shifts, and CoMP coordination decisions.

- **Training** (`CoMP_Tune.py`):
   - Implements PPO with Epsilon-Greedy and Gaussian noise exploration strategies.
   - Optimizes network parameters for maximized energy efficiency and minimized interference.

- **Testing** (`CoMPTest.py`):
   - Loads saved PPO checkpoints and evaluates the trained model over multiple channel realizations.
   - Outputs include rates for edge and near users, outage probabilities, and efficiency metrics.

## Requirements

- Python 3.11+
- Ray and RLlib for reinforcement learning
- Gymnasium for environment management
- [`comyx`](https://comyx.readthedocs.io/latest/) library for network modeling
- TensorBoard for logging and visualization (optional)
