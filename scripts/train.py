#!/usr/bin/env python3
"""
Training script for PLM-RiPP classifier
"""
import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from plm_ripp.utils import setup_seed
from plm_ripp.train import setup_training, training_loop


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for training."""
    parser = argparse.ArgumentParser(
        description="Train ESM-C sequence classifier with LDAM loss",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data arguments
    parser.add_argument("--train_csv", required=True, help="Path to training CSV")
    parser.add_argument("--valid_csv", required=True, help="Path to validation CSV")
    parser.add_argument("--label_names", nargs="+", default=None,
                        help="Optional class names for reporting")

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=10, help="Maximum epochs")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Max gradient norm for clipping")
    parser.add_argument("--weight_decay", type=float, default=1e-2,
                        help="Weight decay")

    # Model arguments
    parser.add_argument("--esm_model", default="esmc_600m",
                        help="ESM-C model variant")
    parser.add_argument("--d_model", type=int, default=1152,
                        help="Hidden dimension size")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout probability")
    parser.add_argument("--freeze_esm", action="store_true", default=True,
                        help="Freeze ESM-C parameters")
    parser.add_argument("--esm_embed_batch_size", type=int, default=0,
                        help="ESM embedding batch size (0=auto)")

    # Loss arguments
    parser.add_argument("--focal_gamma", type=float, default=2.0,
                        help="Deprecated. Kept for compatibility.")
    parser.add_argument("--ldam_max_m", type=float, default=0.5,
                        help="LDAM maximum margin")
    parser.add_argument("--ldam_s", type=float, default=30.0,
                        help="LDAM logit scaling factor")
    parser.add_argument("--ldam_use_class_weight", action="store_true", default=False,
                        help="Use inverse-frequency class weights in LDAM")

    # Output arguments
    parser.add_argument("--save_path", default="best_model.pt",
                        help="Path to save best model checkpoint")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser


def main():
    """Main training function."""
    parser = build_parser()
    args = parser.parse_args()

    # Set random seed
    setup_seed(args.seed)

    # Setup training components
    model, train_loader, valid_loader, optimizer, scheduler, label_names = setup_training(
        train_csv=args.train_csv,
        valid_csv=args.valid_csv,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_labels=0,  # Auto-detect
        esm_model=args.esm_model,
        d_model=args.d_model,
        dropout=args.dropout,
        freeze_esm=args.freeze_esm,
        focal_gamma=args.focal_gamma,
        ldam_max_m=args.ldam_max_m,
        ldam_s=args.ldam_s,
        ldam_use_class_weight=args.ldam_use_class_weight,
        esm_embed_batch_size=args.esm_embed_batch_size,
        device=args.device,
        label_names=args.label_names,
    )

    # Run training loop
    training_loop(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.num_epochs,
        patience=args.patience,
        max_grad_norm=args.max_grad_norm,
        save_path=args.save_path,
        label_names=label_names,
    )


if __name__ == "__main__":
    main()
