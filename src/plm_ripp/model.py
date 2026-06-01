"""
ESM-C based sequence classifier
"""
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from esm.models.esmc import ESMC, _BatchedESMProteinTensor
from esm.sdk.api import LogitsConfig

from .loss import LDAMLoss


class ESMCClassifier(nn.Module):
    """
    Protein sequence classifier using ESM-C embeddings.

    Architecture:
        ESM-C embeddings → LayerNorm → FFN → LayerNorm → 2-layer Bi-LSTM
        → (MaxPool + MeanPool + CLS) concatenation → Classification head

    Args:
        num_labels: Number of output classes
        esm_model_name: ESM-C model variant (e.g., "esmc_600m")
        d_model: Hidden dimension size
        dropout: Dropout probability
        freeze_esm: Whether to freeze ESM-C parameters
        focal_gamma: Deprecated, kept for backward compatibility
        ldam_max_m: LDAM maximum margin
        ldam_s: LDAM logit scaling factor
        esm_embed_batch_size: Batch size for ESM embedding extraction (0=auto)
        device: Device to run model on
    """

    def __init__(
        self,
        num_labels: int,
        esm_model_name: str = "esmc_600m",
        d_model: int = 1152,
        dropout: float = 0.3,
        freeze_esm: bool = True,
        focal_gamma: float = 2.0,
        ldam_max_m: float = 0.5,
        ldam_s: float = 30.0,
        esm_embed_batch_size: int = 0,
        device: str = "cuda",
    ):
        super().__init__()
        self._hparams = dict(
            num_labels=num_labels,
            esm_model_name=esm_model_name,
            d_model=d_model,
            dropout=dropout,
            freeze_esm=freeze_esm,
            focal_gamma=focal_gamma,
            ldam_max_m=ldam_max_m,
            ldam_s=ldam_s,
            esm_embed_batch_size=esm_embed_batch_size,
        )

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.num_labels = num_labels
        self.d_model = d_model
        self.freeze_esm = freeze_esm
        self.focal_gamma = focal_gamma
        self.ldam_max_m = float(ldam_max_m)
        self.ldam_s = float(ldam_s)
        self.esm_embed_batch_size = max(0, int(esm_embed_batch_size))

        print(f"[Model] Loading ESM-C: {esm_model_name} on {device}")
        self.esm = ESMC.from_pretrained(esm_model_name, device=self.device)
        if freeze_esm:
            for p in self.esm.parameters():
                p.requires_grad = False
            print("[Model] ESM-C parameters frozen")
        else:
            print("[Model] ESM-C parameters trainable")

        # Post-embedding processing
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model // 2,
            num_layers=2,
            dropout=dropout,
            bidirectional=True,
            batch_first=True,
        )

        # Classification head (3*d_model due to max+mean+cls pooling)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_labels),
        )

        self.criterion = nn.CrossEntropyLoss()
        self.to(self.device)

    def set_ldam_criterion(
        self,
        cls_num_list: List[int],
        class_weight: Optional[torch.Tensor] = None
    ):
        """
        Set LDAM loss as the training criterion.

        Args:
            cls_num_list: List of sample counts per class
            class_weight: Optional per-class weights
        """
        if class_weight is not None:
            class_weight = class_weight.to(self.device)
        self.criterion = LDAMLoss(
            cls_num_list=cls_num_list,
            max_m=self.ldam_max_m,
            s=self.ldam_s,
            weight=class_weight,
        ).to(self.device)

    def get_config(self) -> dict:
        """Get model hyperparameters."""
        return dict(self._hparams)

    def save_checkpoint(self, path: str, extra: Optional[dict] = None):
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint
            extra: Optional extra metadata to save
        """
        payload = {"config": self.get_config(), "state_dict": self.state_dict()}
        if extra:
            payload.update(extra)
        torch.save(payload, path)

    @classmethod
    def load_checkpoint(cls, path: str, device: str = "cuda") -> "ESMCClassifier":
        """
        Load model from checkpoint.

        Args:
            path: Path to checkpoint file
            device: Device to load model on

        Returns:
            Loaded model instance
        """
        ckpt = torch.load(path, map_location="cpu")
        config = ckpt["config"]
        config["device"] = device
        model = cls(**config)
        model.load_state_dict(ckpt["state_dict"], strict=False)
        model.eval()
        best_metric = ckpt.get("val_f1", ckpt.get("val_acc", "?"))
        print(
            f"[Model] Loaded from {path} (epoch={ckpt.get('epoch', '?')}, "
            f"best_val_f1={best_metric})"
        )
        return model

    def _extract_esmc_embeddings(
        self, sequences: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract ESM-C embeddings for a batch of sequences.

        Args:
            sequences: List of protein sequences

        Returns:
            Tuple of (embeddings [B, L, D], mask [B, L])
        """
        lengths = [len(seq) for seq in sequences]
        if not lengths:
            raise ValueError("sequences must not be empty")

        B, Lmax = len(sequences), max(lengths)
        ctx = torch.no_grad() if self.freeze_esm else torch.enable_grad()
        chunk_size = (
            int(self.esm_embed_batch_size)
            if int(self.esm_embed_batch_size) > 0
            else len(sequences)
        )
        chunk_size = max(1, chunk_size)

        esm_batch = None
        esm_mask = None

        with ctx:
            for i in range(0, len(sequences), chunk_size):
                seq_chunk = sequences[i : i + chunk_size]
                token_batch = self.esm._tokenize(seq_chunk)
                logits_out = self.esm.logits(
                    _BatchedESMProteinTensor(sequence=token_batch),
                    LogitsConfig(sequence=True, return_embeddings=True),
                )
                emb_chunk = logits_out.embeddings
                if emb_chunk is None:
                    raise RuntimeError(
                        "ESM-C embeddings is None. Please check LogitsConfig."
                    )
                if emb_chunk.dim() == 2:
                    emb_chunk = emb_chunk.unsqueeze(0)

                if esm_batch is None:
                    D = emb_chunk.size(-1)
                    esm_batch = torch.zeros(
                        B, Lmax, D, device=self.device, dtype=emb_chunk.dtype
                    )
                    esm_mask = torch.zeros(B, Lmax, device=self.device, dtype=torch.long)

                for local_idx, (seq, emb) in enumerate(zip(seq_chunk, emb_chunk)):
                    global_idx = i + local_idx
                    seq_len = len(seq)
                    if emb.size(0) > seq_len:
                        emb = emb[:seq_len]
                    elif emb.size(0) < seq_len:
                        pad = torch.zeros(
                            seq_len - emb.size(0),
                            emb.size(1),
                            device=emb.device,
                            dtype=emb.dtype,
                        )
                        emb = torch.cat([emb, pad], dim=0)
                    esm_batch[global_idx, :seq_len] = emb
                    esm_mask[global_idx, :seq_len] = 1

        return esm_batch, esm_mask

    def concat_pooling(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Apply concatenated pooling: max + mean + CLS.

        Args:
            x: Sequence representations [B, L, D]
            mask: Valid token mask [B, L]

        Returns:
            Pooled representation [B, 3*D]
        """
        mask_f = mask.unsqueeze(-1).float()  # [B, L, 1]

        # Max pooling over valid tokens
        x_max = x.masked_fill(mask.unsqueeze(-1) == 0, -1e9).max(dim=1).values

        # Mean pooling over valid tokens
        denom = mask_f.sum(dim=1).clamp_min(1.0)
        x_mean = (x * mask_f).sum(dim=1) / denom

        # CLS pooling (first token)
        x_cls = x[:, 0, :]

        return torch.cat([x_max, x_mean, x_cls], dim=-1)

    def forward(
        self, sequences: List[str], labels: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Forward pass.

        Args:
            sequences: List of protein sequences
            labels: Optional ground truth labels

        Returns:
            Dictionary with 'logits' and optionally 'loss'
        """
        esm_emb, esm_mask = self._extract_esmc_embeddings(sequences)

        x = self.norm1(esm_emb)
        x = self.norm2(x + self.dropout(self.ffn(x)))
        x, _ = self.lstm(x)

        pooled = self.concat_pooling(x, esm_mask)
        logits = self.classifier(self.dropout(pooled))

        output = {"logits": logits}
        if labels is not None:
            labels = labels.to(self.device)
            output["loss"] = self.criterion(logits, labels)
        return output
