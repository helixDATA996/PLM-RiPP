"""
PLM-RiPP: Protein Language Model for RiPP Classification

A deep learning framework for classifying Ribosomally synthesized and
Post-translationally modified Peptides (RiPPs) using ESM-C embeddings.
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from .model import ESMCClassifier
from .dataset import SequenceClassificationDataset
from .loss import LDAMLoss
from .metrics import compute_macro_f1, compute_class_weights

__all__ = [
    "ESMCClassifier",
    "SequenceClassificationDataset",
    "LDAMLoss",
    "compute_macro_f1",
    "compute_class_weights",
]
