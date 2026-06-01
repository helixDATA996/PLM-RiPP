"""
Dataset and data loading utilities
"""
from typing import List, Dict

import pandas as pd
import torch
import torch.utils.data as data


class SequenceClassificationDataset(data.Dataset):
    """
    Dataset for protein sequence classification.

    Loads sequences and labels from a CSV file with columns:
    - sequence: Protein sequence string
    - label: Integer class label

    Args:
        csv_path: Path to CSV file
    """

    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        if "sequence" not in self.df.columns or "label" not in self.df.columns:
            raise ValueError("CSV must include columns: sequence, label")

        self.sequences = self.df["sequence"].astype(str).tolist()
        self.labels = self.df["label"].astype(int).tolist()

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx: int):
        return {"sequence": self.sequences[idx], "label": self.labels[idx]}


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Collate function for DataLoader.

    Args:
        batch: List of samples from dataset

    Returns:
        Dictionary with batched sequences and labels
    """
    sequences = [x["sequence"] for x in batch]
    labels = torch.tensor([x["label"] for x in batch], dtype=torch.long)
    return {"sequences": sequences, "labels": labels}
