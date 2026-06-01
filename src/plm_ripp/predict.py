"""
Prediction utilities
"""
import os
import math
from typing import List, Optional

import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import classification_report

from .model import ESMCClassifier
from .utils import parse_fasta


@torch.inference_mode()
def predict_batch(
    model: ESMCClassifier,
    input_path: str,
    output_csv: str,
    batch_size: int = 8,
    label_names: Optional[List[str]] = None,
    clear_cuda_cache_each_batch: bool = False,
) -> pd.DataFrame:
    """
    Batch prediction on sequences from FASTA or CSV file.

    Args:
        model: Trained model
        input_path: Path to input file (FASTA or CSV)
        output_csv: Path to save predictions
        batch_size: Batch size for inference
        label_names: Optional class names
        clear_cuda_cache_each_batch: Whether to clear CUDA cache after each batch

    Returns:
        DataFrame with predictions and probabilities
    """
    model.eval()
    ext = os.path.splitext(input_path)[1].lower()

    # Load sequences
    if ext in (".fa", ".fasta", ".fna"):
        ids, sequences = parse_fasta(input_path)
        df = pd.DataFrame({"id": ids, "sequence": sequences})
        print(f"[Predict] Loaded FASTA: {len(sequences)} sequences")
    else:
        df = pd.read_csv(input_path)
        if "sequence" not in df.columns:
            raise ValueError("CSV input must contain 'sequence' column")
        sequences = df["sequence"].astype(str).tolist()
        print(f"[Predict] Loaded CSV: {len(sequences)} sequences")

    all_preds, all_probs = [], []
    total = len(sequences)

    pbar = tqdm(
        range(0, total, batch_size),
        desc="[Predict]",
        dynamic_ncols=True,
        total=math.ceil(total / batch_size),
        unit="batch",
    )
    for start in pbar:
        end = min(start + batch_size, total)
        batch_seqs = sequences[start:end]
        out = model(sequences=batch_seqs)
        probs = torch.softmax(out["logits"], dim=-1)
        preds = torch.argmax(probs, dim=-1)

        all_preds.extend(preds.cpu().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())
        pbar.set_postfix(done=f"{end}/{total}")

        del out, probs, preds
        if clear_cuda_cache_each_batch and torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Add predictions to dataframe
    df["predicted_label"] = all_preds
    if label_names is not None:
        df["predicted_name"] = [label_names[p] for p in all_preds]

    # Add probability columns
    num_classes = len(all_probs[0])
    for c in range(num_classes):
        col = f"prob_class_{c}" if label_names is None else f"prob_{label_names[c]}"
        df[col] = [p[c] for p in all_probs]

    # If ground truth labels exist, compute metrics
    if "label" in df.columns:
        true_labels = df["label"].astype(int).tolist()
        acc = sum(p == t for p, t in zip(all_preds, true_labels)) / total
        print(f"\n[Predict] Accuracy: {acc:.4f}")
        n = model.num_labels
        tgt = label_names if label_names and len(label_names) == n else None
        print(
            classification_report(
                true_labels,
                all_preds,
                labels=list(range(n)),
                target_names=tgt,
                zero_division=0,
            )
        )

    df.to_csv(output_csv, index=False)
    print(f"[Predict] Results saved to: {output_csv}")
    return df
