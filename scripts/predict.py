#!/usr/bin/env python3
"""
Prediction script for PLM-RiPP classifier
"""
import argparse
import sys
import os

import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from plm_ripp.model import ESMCClassifier
from plm_ripp.predict import predict_batch


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for prediction."""
    parser = argparse.ArgumentParser(
        description="Batch prediction with trained ESM-C classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--checkpoint", required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--input", required=True,
                        help="Input file (FASTA or CSV with 'sequence' column)")
    parser.add_argument("--output_csv", default="predictions.csv",
                        help="Output CSV path")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for inference")
    parser.add_argument("--label_names", nargs="+", default=None,
                        help="Optional class names (overrides checkpoint)")
    parser.add_argument("--esm_embed_batch_size", type=int, default=None,
                        help="Override ESM embedding batch size from checkpoint")
    parser.add_argument("--clear_cuda_cache_each_batch", action="store_true",
                        default=False,
                        help="Clear CUDA cache after each batch (slower but saves memory)")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")

    return parser


def main():
    """Main prediction function."""
    parser = build_parser()
    args = parser.parse_args()

    # Load model
    device = args.device if torch.cuda.is_available() else "cpu"
    model = ESMCClassifier.load_checkpoint(args.checkpoint, device=device)

    # Override ESM embedding batch size if specified
    if args.esm_embed_batch_size is not None:
        model.esm_embed_batch_size = int(args.esm_embed_batch_size)
        print(f"[Predict] esm_embed_batch_size={model.esm_embed_batch_size}")

    # Get label names from checkpoint or args
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    label_names = args.label_names or ckpt.get("label_names", None)

    # Run prediction
    predict_batch(
        model=model,
        input_path=args.input,
        output_csv=args.output_csv,
        batch_size=args.batch_size,
        label_names=label_names,
        clear_cuda_cache_each_batch=args.clear_cuda_cache_each_batch,
    )


if __name__ == "__main__":
    main()
