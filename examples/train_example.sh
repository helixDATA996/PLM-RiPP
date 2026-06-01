#!/bin/bash
# Training script based on your original parameters
# Updated for the refactored project structure

nohup python scripts/train.py \
  --train_csv data/train.csv \
  --valid_csv data/valid.csv \
  --label_names NEGATIVE AUTO_INDUCING_PEPTIDE BACTERIAL_HEAD_TO_TAIL_CYCLIZED CYANOBACTIN LANTHIPEPTIDE LASSO_PEPTIDE rSAM_MODIFIED_RiPP THIOPEPTIDE OEP OTHER \
  --batch_size 48 \
  --esm_embed_batch_size 48 \
  --lr 1e-4 \
  --num_epochs 50 \
  --patience 10 \
  --d_model 1152 \
  --dropout 0.2 \
  --weight_decay 1e-2 \
  --ldam_max_m 0.5 \
  --ldam_s 20 \
  --ldam_use_class_weight \
  --freeze_esm \
  --esm_model esmc_600m \
  --device cuda \
  --save_path models/best_model_ripp_refactored.pt \
  --seed 42 \
  > train_refactored.log 2>&1 &

echo "Training started in background. Check train_refactored.log for progress."
echo "Use 'tail -f train_refactored.log' to monitor training."
