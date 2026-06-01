# PLM-RiPP: Protein Language Model for RiPP Classification

A deep learning framework for classifying Ribosomally synthesized and Post-translationally modified Peptides (RiPPs) using ESM-C protein language model embeddings.

## Features

- **ESM-C Embeddings**: Leverages state-of-the-art protein language model (ESM-C) for sequence representation
- **Advanced Architecture**: Combines LayerNorm, FFN, Bi-LSTM, and multi-pooling strategies (max, mean, CLS)
- **Class Imbalance Handling**: Implements LDAM (Label-Distribution-Aware Margin) loss for long-tailed recognition
- **Flexible Training**: Supports frozen or fine-tuned ESM-C backbone
- **Batch Prediction**: Efficient inference on FASTA or CSV files with probability outputs

## Architecture

```
Input Sequences
    ↓
ESM-C Embeddings
    ↓
LayerNorm → FFN → LayerNorm
    ↓
2-layer Bidirectional LSTM
    ↓
Concatenated Pooling (Max + Mean + CLS)
    ↓
Classification Head (3*d_model → 512 → 256 → num_classes)
```

## Installation

### Requirements

- Python >= 3.8
- PyTorch >= 2.1 with CUDA support (recommended)
- ESM >= 3.0.0

### Setup

```bash
# Clone the repository
git clone https://github.com/helixDATA996/PLM-RiPP.git
cd PLM-RiPP

# Install PyTorch (choose one according to your system)
# For CUDA 12.4 (recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# For CPU only
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install -r requirements.txt

```

## Quick Start

### Training

```bash
python scripts/train.py \
    --train_csv data/train.csv \
    --valid_csv data/valid.csv \
    --batch_size 48 \
    --lr 1e-4 \
    --num_epochs 50 \
    --save_path models/best_model.pt \
    --device cuda
```

### Prediction
First, download the pretrained model checkpoint and place it in the `models/` directory.

📥 [Download checkpoint](https://drive.google.com/drive/folders/1J9bAadIiJmuFiZuqmivWgv7DbwbieznW?usp=drive_link)

```bash
# Predict on FASTA file
python scripts/predict.py \
    --checkpoint models/best_model_ripp.pt \
    --input sequences.fasta \
    --output_csv predictions.csv \
    --device cuda

# Predict on CSV file (must have 'sequence' column)
python scripts/predict.py \
    --checkpoint models/best_model_ripp.pt \
    --input test.csv \
    --output_csv predictions.csv \
    --device cuda
```

## Data Format

### Training Data (CSV)

```csv
sequence,label
METLVNLFFKFFTSIMEFVGLVAGANPCAGYFDEPEVPDELTKLYE,1
MKNILKILSLKFTSNICTRMALSVSASACHWSAYQPEEPKCLRDIKNH,1
MFKRFKTQILSSLASFVAVVAFSGVSATSMWIFYEPDIPKALKDK,1
```

- `sequence`: Protein sequence (amino acid string)
- `label`: Integer class label (0, 1, 2, ...)

### Prediction Input

**FASTA format:**
```
>seq1
METLVNLFFKFFTSIMEFVGLVAGANPCAGYFDEPEVPDELTKLYE
>seq2
MKNILKILSLKFTSNICTRMALSVSASACHWSAYQPEEPKCLRDIKNH
```

**CSV format:**
```csv
id,sequence
seq1,METLVNLFFKFFTSIMEFVGLVAGANPCAGYFDEPEVPDELTKLYE
seq2,MKNILKILSLKFTSNICTRMALSVSASACHWSAYQPEEPKCLRDIKNH
```

## Project Structure

```
PLM-RiPP/
├── README.md                    # This file
├── LICENSE                      # License file
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
├── data/                        # Data directory
│   ├── train.csv               # Training data
│   ├── valid.csv               # Validation data
│   └── README.md               # Data documentation
├── models/                      # Model checkpoints
│   ├── best_model_ripp.pt      # Pre-trained model
│   └── README.md               # Model documentation
├── src/                         # Source code
│   └── plm_ripp/
│       ├── __init__.py         # Package initialization
│       ├── model.py            # ESMCClassifier model
│       ├── dataset.py          # Dataset and data loading
│       ├── loss.py             # LDAM loss function
│       ├── metrics.py          # Evaluation metrics
│       ├── train.py            # Training logic
│       ├── predict.py          # Prediction logic
│       └── utils.py            # Utility functions
├── scripts/                     # Executable scripts
│   ├── train.py                # Training entry point
│   └── predict.py              # Prediction entry point
└── examples/                    # Usage examples
    ├── train_example.sh        # Training example
    └── predict_example.sh      # Prediction example
```

## Advanced Usage

### Custom Training Parameters

```bash
python scripts/train.py \
    --train_csv data/train.csv \
    --valid_csv data/valid.csv \
    --batch_size 16 \
    --lr 5e-5 \
    --num_epochs 20 \
    --patience 5 \
    --d_model 1152 \
    --dropout 0.3 \
    --freeze_esm \
    --ldam_max_m 0.5 \
    --ldam_s 30.0 \
    --ldam_use_class_weight \
    --esm_model esmc_600m \
    --save_path models/custom_model.pt \
    --device cuda \
    --label_names Lanthipeptide Thiopeptide Lassopeptide
```

### Memory-Efficient Prediction

For large datasets or limited GPU memory:

```bash
python scripts/predict.py \
    --checkpoint models/best_model_ripp.pt \
    --input large_dataset.fasta \
    --output_csv predictions.csv \
    --batch_size 4 \
    --esm_embed_batch_size 2 \
    --clear_cuda_cache_each_batch \
    --device cuda
```

## Model Details

### Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `d_model` | 1152 | Hidden dimension size |
| `dropout` | 0.2 | Dropout probability |
| `freeze_esm` | True | Freeze ESM-C parameters |
| `ldam_max_m` | 0.5 | LDAM maximum margin |
| `ldam_s` | 20.0 | LDAM logit scaling factor |
| `lr` | 1e-4 | Learning rate |
| `batch_size` | 48 | Training batch size |

### Loss Function

The model uses **LDAM (Label-Distribution-Aware Margin) Loss** to handle class imbalance:

- Applies class-dependent margins inversely proportional to class frequency
- Helps balance decision boundaries for minority classes
- Reference: Cao et al. "Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss" (NeurIPS 2019)

### Evaluation Metric

- **Primary metric**: Macro-averaged F1 score
- Gives equal weight to all classes regardless of support
- Used for model selection and early stopping

## Citation

If you use this code in your research, please cite:

```bibtex


```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- ESM-C model from [EvolutionaryScale](https://github.com/evolutionaryscale/esm)
- LDAM loss implementation based on [Cao et al. (NeurIPS 2019)](https://arxiv.org/abs/1906.07413)

## Contact

For questions or issues, please open an issue on GitHub or contact [1351771272@qq.com].
