"""
MuJoCo-Bench-IDO Baselines Module
===================================

Provides adapter classes for baseline RL agents and world models,
enabling comparative evaluation against the IDO/TOMAS agent.

Available baselines:
  - TDMPC2Adapter: TD-MPC2 model-based RL baseline (control comparison)
  - CosmosPredictAdapter: NVIDIA Cosmos-Predict1 world model (η trajectory prediction)

Both adapters implement graceful degradation: if the required package
is not installed, the adapter prints a warning and returns None, allowing
the benchmark framework to skip that baseline gracefully.

Author: MuJoCo-Bench-IDO v0.3.0 baseline integration
"""

from baselines.tdmpc2_adapter import TDMPC2Adapter, make_tdmpc2_adapter
from baselines.cosmos_predict_adapter import CosmosPredictAdapter, make_cosmos_predict_adapter

__all__ = [
    'TDMPC2Adapter',
    'CosmosPredictAdapter',
    'make_tdmpc2_adapter',
    'make_cosmos_predict_adapter',
]
