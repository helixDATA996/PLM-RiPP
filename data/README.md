# Data Directory

This directory contains training and validation data for the PLM-RiPP classifier.

## Files

- `train.csv`: Training dataset (44,813 sequences)
- `valid.csv`: Validation dataset (4,690 sequences)

## Data Format

Both CSV files follow the same format:

```csv
sequence,label
METLVNLFFKFFTSIMEFVGLVAGANPCAGYFDEPEVPDELTKLYE,1
MKNILKILSLKFTSNICTRMALSVSASACHWSAYQPEEPKCLRDIKNH,1
```

### Columns

- **sequence**: Protein sequence as a string of amino acids
- **label**: Integer class label (0, 1, 2, ...)

## Usage

These files are used by the training script:

```bash
python scripts/train.py \
    --train_csv data/train.csv \
    --valid_csv data/valid.csv \
    --save_path models/best_model.pt
```

## Adding Your Own Data

To train on your own data:

1. Prepare CSV files with `sequence` and `label` columns
2. Ensure labels are integers starting from 0
3. Split your data into training and validation sets
4. Update the paths in your training command

## Data Statistics

- **Training samples**: 44,813
- **Validation samples**: 4,690
- **Total samples**: 49,503
- **Classes**: Multiple (auto-detected during training)

## Notes

- Sequences should contain only standard amino acid letters
- Class labels should be consecutive integers (0, 1, 2, ...)
- The model automatically detects the number of classes from the data
