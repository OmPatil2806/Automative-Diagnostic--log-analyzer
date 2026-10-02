"""Synthetic fault log generation."""

from analyzer.synthetic.faults import FAULTS, Fault
from analyzer.synthetic.generator import SyntheticDataset, generate, inject_fault

__all__ = ["FAULTS", "Fault", "SyntheticDataset", "generate", "inject_fault"]
