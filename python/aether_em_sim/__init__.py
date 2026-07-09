"""Electromagnetic Field Data Simulator.

Pure-Python reference utilities for generating Faraday-tensor coupling datasets.
"""

from .core import FaradayTensor, ScenarioConfig, Vec3, simulate_scenario

__all__ = ["FaradayTensor", "ScenarioConfig", "Vec3", "simulate_scenario"]
