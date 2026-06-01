#!/bin/bash
# Example prediction script for PLM-RiPP

# Predict on a FASTA file
python scripts/predict.py \
    --checkpoint models/best_model_ripp.pt \
    --input data/test_sequences.fasta \
    --output_csv predictions.csv \
    --batch_size 8 \
    --device cuda

# Predict on a CSV file (must have 'sequence' column)
# python scripts/predict.py \
#     --checkpoint models/best_model_ripp.pt \
#     --input data/test.csv \
#     --output_csv predictions.csv \
#     --batch_size 8 \
#     --device cuda

# Optional: Override label names
# --label_names Class0 Class1 Class2 ...

# Optional: For memory-constrained GPUs
# --clear_cuda_cache_each_batch \
# --esm_embed_batch_size 4
