"""
MuJoCo-Bench-IDO Baselines Module
===================================

Provides adapter classes for baseline RL agents and world models,
enabling comparative evaluation against the IDO/TOMAS agent.

Available baselines:
  - SB3PPOAdapter: Stable-Baselines3 PPO baseline (actual policy actions)
  - SB3SACAdapter: Stable-Baselines3 SAC baseline (actual policy actions)
  - TDMPC2Adapter: TD-MPC2 model-based RL baseline (control comparison)
  - DreamerV3Adapter: DreamerV3 model-based RL (SOTA motor layer)
  - CosmosPredictAdapter: NVIDIA Cosmos-Predict1 world model (η trajectory prediction)

All adapters implement graceful degradation: if the required package
is not installed, the adapter prints a warning and falls back to random
actions, allowing the benchmark framework to continue gracefully.

Author: MuJoCo-Bench-IDO v0.9.0 DreamerV3 integration
"""

from baselines.sb3_adapter import (
    SB3PPOAdapter,
    SB3SACAdapter,
    make_sb3_ppo_adapter,
    make_sb3_sac_adapter,
)
from baselines.tdmpc2_adapter import TDMPC2Adapter, make_tdmpc2_adapter
from baselines.dreamer_adapter import DreamerV3Adapter, make_dreamer_adapter
from baselines.cosmos_predict_adapter import CosmosPredictAdapter, make_cosmos_predict_adapter

__all__ = [
    'SB3PPOAdapter',
    'SB3SACAdapter',
    'TDMPC2Adapter',
    'DreamerV3Adapter',
    'CosmosPredictAdapter',
    'make_sb3_ppo_adapter',
    'make_sb3_sac_adapter',
    'make_tdmpc2_adapter',
    'make_dreamer_adapter',
    'make_cosmos_predict_adapter',
]
