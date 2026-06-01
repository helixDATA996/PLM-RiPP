"""
Utility functions for PLM-RiPP
"""
import random
from typing import List, Tuple

import numpy as np
import torch


def setup_seed(seed: int = 42):
    """
    Set random seeds for reproducibility across all libraries.

    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def parse_fasta(path: str) -> Tuple[List[str], List[str]]:
    """
    Parse a FASTA file and extract sequence IDs and sequences.

    Args:
        path: Path to FASTA file

    Returns:
        Tuple of (sequence_ids, sequences)
    """
    ids, seqs = [], []
    cur_id, cur_seq = None, []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur_id is not None:
                    ids.append(cur_id)
                    seqs.append("".join(cur_seq))
                cur_id = line[1:].split()[0]
                cur_seq = []
            else:
                cur_seq.append(line)

    if cur_id is not None:
        ids.append(cur_id)
        seqs.append("".join(cur_seq))

    return ids, seqs
