"""
Loss functions for handling class imbalance
"""
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LDAMLoss(nn.Module):
    """
    Label-Distribution-Aware Margin (LDAM) Loss for long-tailed recognition.

    Applies class-dependent margins to logits before computing cross-entropy loss.
    The margin for each class is inversely proportional to the fourth root of
    its sample count, helping to balance the decision boundaries.

    Reference:
        Cao et al. "Learning Imbalanced Datasets with Label-Distribution-Aware
        Margin Loss" (NeurIPS 2019)

    Args:
        cls_num_list: List of sample counts for each class
        max_m: Maximum margin value
        s: Scaling factor for logits
        weight: Optional per-class weights for cross-entropy loss
    """

    def __init__(
        self,
        cls_num_list: List[int],
        max_m: float = 0.5,
        s: float = 30.0,
        weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        cls_arr = np.array(cls_num_list, dtype=np.float32)
        if np.any(cls_arr <= 0):
            raise ValueError(
                f"cls_num_list must be >0 for all classes, got: {cls_num_list}"
            )

        # Compute per-class margins: m_y = C / (n_y^(1/4))
        m_list = 1.0 / np.sqrt(np.sqrt(cls_arr))
        m_list = m_list * (max_m / np.max(m_list))

        self.register_buffer("m_list", torch.tensor(m_list, dtype=torch.float32))
        self.s = float(s)
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute LDAM loss.

        Args:
            logits: Model predictions [batch_size, num_classes]
            targets: Ground truth labels [batch_size]

        Returns:
            Scalar loss value
        """
        if logits.ndim != 2:
            raise ValueError(
                f"logits must be [B, C], got shape={tuple(logits.shape)}"
            )
        if targets.ndim != 1:
            raise ValueError(
                f"targets must be [B], got shape={tuple(targets.shape)}"
            )

        batch_size = targets.size(0)
        margins = self.m_list.to(logits.device)[targets]  # [B]

        # Apply margin: logits_y <- logits_y - m_y
        logits_adj = logits.clone()
        logits_adj[torch.arange(batch_size, device=logits.device), targets] -= margins

        # Scale and compute cross-entropy
        return F.cross_entropy(self.s * logits_adj, targets, weight=self.weight)
