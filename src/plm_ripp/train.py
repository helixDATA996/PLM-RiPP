"""
Training logic and utilities
"""
from typing import Tuple, List, Optional
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from tqdm import tqdm
from sklearn.metrics import classification_report

from .model import ESMCClassifier
from .dataset import SequenceClassificationDataset, collate_fn
from .metrics import compute_macro_f1, compute_class_weights


def train_one_epoch(
    model: ESMCClassifier,
    loader: data.DataLoader,
    optimizer: optim.Optimizer,
    epoch: int,
    max_grad_norm: float = 1.0,
) -> Tuple[float, float]:
    """
    Train model for one epoch.

    Args:
        model: Model to train
        loader: Training data loader
        optimizer: Optimizer
        epoch: Current epoch number
        max_grad_norm: Maximum gradient norm for clipping

    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()
    total_loss, total, correct = 0.0, 0, 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", dynamic_ncols=True)
    for batch in pbar:
        sequences = batch["sequences"]
        labels = batch["labels"].to(model.device)

        optimizer.zero_grad()
        out = model(sequences=sequences, labels=labels)
        loss = out["loss"]
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        bs = labels.size(0)
        preds = torch.argmax(out["logits"], dim=-1)
        total_loss += loss.item() * bs
        correct += (preds == labels).sum().item()
        total += bs

        pbar.set_postfix(loss=f"{total_loss/total:.4f}", acc=f"{correct/total:.4f}")

    return total_loss / total, correct / total


@torch.inference_mode()
def evaluate(
    model: ESMCClassifier,
    loader: data.DataLoader
) -> Tuple[float, float, float, List[int], List[int]]:
    """
    Evaluate model on validation/test set.

    Args:
        model: Model to evaluate
        loader: Validation data loader

    Returns:
        Tuple of (loss, accuracy, macro_f1, predictions, labels)
    """
    model.eval()
    total_loss, total, correct = 0.0, 0, 0
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc=f"         [Valid]", dynamic_ncols=True)
    for batch in pbar:
        sequences = batch["sequences"]
        labels = batch["labels"].to(model.device)
        out = model(sequences=sequences, labels=labels)

        bs = labels.size(0)
        preds = torch.argmax(out["logits"], dim=-1)
        total_loss += out["loss"].item() * bs
        correct += (preds == labels).sum().item()
        total += bs
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

        pbar.set_postfix(loss=f"{total_loss/total:.4f}", acc=f"{correct/total:.4f}")

    val_acc = correct / total
    val_f1 = compute_macro_f1(all_labels, all_preds, model.num_labels)
    return total_loss / total, val_acc, val_f1, all_preds, all_labels


def training_loop(
    model: ESMCClassifier,
    train_loader: data.DataLoader,
    valid_loader: data.DataLoader,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    num_epochs: int,
    patience: int,
    max_grad_norm: float,
    save_path: str,
    label_names: Optional[List[str]] = None,
):
    """
    Main training loop with early stopping.

    Args:
        model: Model to train
        train_loader: Training data loader
        valid_loader: Validation data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        num_epochs: Maximum number of epochs
        patience: Early stopping patience
        max_grad_norm: Maximum gradient norm for clipping
        save_path: Path to save best model
        label_names: Optional class names for reporting
    """
    best_val_f1, patience_counter = 0.0, 0

    for epoch in range(1, num_epochs + 1):
        print(f"\n===== Epoch {epoch}/{num_epochs} =====")
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, epoch, max_grad_norm
        )
        val_loss, val_acc, val_f1, val_preds, val_labels = evaluate(
            model, valid_loader
        )

        print(f"\n{'─'*45}")
        print(f"  {'':10s}  {'loss':>8s}  {'acc':>8s}  {'macro_f1':>8s}")
        print(f"  {'Train':10s}  {train_loss:8.4f}  {train_acc:8.4f}  {'-':>8s}")
        print(f"  {'Valid':10s}  {val_loss:8.4f}  {val_acc:8.4f}  {val_f1:8.4f}")

        n = model.num_labels
        tgt = label_names if label_names and len(label_names) == n else None
        print(
            classification_report(
                val_labels,
                val_preds,
                labels=list(range(n)),
                target_names=tgt,
                zero_division=0,
            )
        )
        print(f"{'─'*45}")

        scheduler.step(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1, patience_counter = val_f1, 0
            model.save_checkpoint(
                save_path,
                extra={
                    "epoch": epoch,
                    "val_f1": val_f1,
                    "val_acc": val_acc,
                    "label_names": label_names,
                },
            )
            print(f"[Saved] {save_path}  val_f1={val_f1:.4f}, val_acc={val_acc:.4f}")
        else:
            patience_counter += 1
            print(f"[EarlyStopping] patience {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    print(f"\nTraining complete. Best val_macro_f1 = {best_val_f1:.4f}")


def setup_training(
    train_csv: str,
    valid_csv: str,
    batch_size: int,
    lr: float,
    weight_decay: float,
    num_labels: int,
    esm_model: str,
    d_model: int,
    dropout: float,
    freeze_esm: bool,
    focal_gamma: float,
    ldam_max_m: float,
    ldam_s: float,
    ldam_use_class_weight: bool,
    esm_embed_batch_size: int,
    device: str,
    label_names: Optional[List[str]] = None,
) -> Tuple[ESMCClassifier, data.DataLoader, data.DataLoader, optim.Optimizer, optim.lr_scheduler._LRScheduler, Optional[List[str]]]:
    """
    Setup training components: model, dataloaders, optimizer, scheduler.

    Args:
        train_csv: Path to training CSV
        valid_csv: Path to validation CSV
        batch_size: Batch size
        lr: Learning rate
        weight_decay: Weight decay
        num_labels: Number of classes (if known, else auto-detect)
        esm_model: ESM model name
        d_model: Hidden dimension
        dropout: Dropout rate
        freeze_esm: Whether to freeze ESM
        focal_gamma: Focal loss gamma (deprecated)
        ldam_max_m: LDAM max margin
        ldam_s: LDAM scaling factor
        ldam_use_class_weight: Whether to use class weights
        esm_embed_batch_size: ESM embedding batch size
        device: Device
        label_names: Optional class names

    Returns:
        Tuple of (model, train_loader, valid_loader, optimizer, scheduler, label_names)
    """
    # Load datasets
    train_ds = SequenceClassificationDataset(train_csv)
    valid_ds = SequenceClassificationDataset(valid_csv)

    # Auto-detect number of classes
    detected_num_labels = len(set(train_ds.labels))
    print(f"[Data] Auto-detected {detected_num_labels} classes: {sorted(set(train_ds.labels))}")
    counter = Counter(train_ds.labels)
    for c in sorted(counter):
        print(f"  Class {c}: {counter[c]} samples")

    if label_names and len(label_names) != detected_num_labels:
        print(f"[Warning] label_names count mismatch, ignoring")
        label_names = None

    # Prepare LDAM loss
    cls_counter = Counter(train_ds.labels)
    cls_num_list = [cls_counter.get(c, 0) for c in range(detected_num_labels)]
    print(f"[LDAM] max_m={ldam_max_m}, s={ldam_s}, class_counts={cls_num_list}")

    class_weight = None
    if ldam_use_class_weight:
        class_weight = compute_class_weights(train_ds.labels, detected_num_labels)
        print(f"[LDAM] class_weight={[f'{w:.4f}' for w in class_weight.tolist()]}")

    # Create dataloaders
    train_loader = data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn
    )
    valid_loader = data.DataLoader(
        valid_ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn
    )

    # Create model
    model = ESMCClassifier(
        num_labels=detected_num_labels,
        esm_model_name=esm_model,
        d_model=d_model,
        dropout=dropout,
        freeze_esm=freeze_esm,
        focal_gamma=focal_gamma,
        ldam_max_m=ldam_max_m,
        ldam_s=ldam_s,
        esm_embed_batch_size=esm_embed_batch_size,
        device=device,
    )
    model.set_ldam_criterion(cls_num_list=cls_num_list, class_weight=class_weight)

    # Create optimizer and scheduler
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=2, factor=0.5
    )

    return model, train_loader, valid_loader, optimizer, scheduler, label_names
