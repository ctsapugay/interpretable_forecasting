# Using Custom Datasets with Configuration Files

This guide explains how to use your own time series dataset with the Extended Interpretable Forecasting Model through the `dataset_config.json` configuration file and command-line arguments.

---

## Quick Start

### **Step 1: Prepare Your Data**

Your CSV file should have:
- One date/timestamp column (optional)
- Multiple numeric variable columns
- Header row with column names

**Example CSV:**
```csv
date,temperature,humidity,pressure,wind_speed
2024-01-01 00:00:00,20.5,65.2,1013.2,5.3
2024-01-01 01:00:00,20.1,66.1,1013.5,4.8
2024-01-01 02:00:00,19.8,67.0,1013.8,4.2
...
```

### **Step 2: Edit `dataset_config.json`**
```
{
  "file_path": "path/to/your/data.csv",
  "variable_columns": ["temperature", "humidity", "pressure", "wind_speed"],
  "date_column": "date",
  "num_variables": 4
}
```

### **Step 3: Running `train_extended_model.py`**
There are many oter command-line arguments you may use, but these are the ones relevant to using your own csv file.
```
--config dataset_config.json       # Path to dataset configuration file
--data-path ETT-small/ETTh1.csv    # Override CSV file path from config
--variable-columns HUFL HULL ...   # Override variable columns from config
--input-length 96                  # Length of input sequences (lookback window)
--forecast-horizon 24              # Length of prediction sequences (forecast steps)
--stride 1                         # Step size between samples when creating windows
--train-ratio 0.7                  # Proportion of data for training (70%)
--val-ratio 0.2                    # Proportion of data for validation (20%)
--test-ratio 0.1                   # Proportion of data for testing (10%)
--embed-dim 32                     # Embedding dimension for univariate learners
--hidden-dim 64                    # Hidden dimension for temporal attention
--num-heads 4                      # Number of attention heads for temporal self-attention
--cross-dim 32                     # Dimension for cross-variable attention
--cross-heads 4                    # Number of attention heads for cross-variable attention
--compressed-dim 64                # Dimension after temporal compression
--num-control-points 8             # Number of B-spline control points for forecasting
--dropout 0.1                      # Dropout rate for regularization
--epochs 50                        # Number of training epochs
--batch-size 32                    # Batch size for training
--lr 0.001                         # Learning rate for optimizer
--weight-decay 1e-5                # L2 regularization weight decay
--scheduler                        # Enable learning rate scheduler (ReduceLROnPlateau)
--save-dir training_outputs        # Directory to save training outputs and checkpoints
--save-freq 5                      # Save attention heatmaps every N epochs
--patience 10                      # Early stopping patience (epochs without improvement)
```

#### Here are some examples:
```
# Use different config file
python train_extended_model.py --config my_config.json

# Override data path
python train_extended_model.py --data-path different_data.csv

# Override variable columns
python train_extended_model.py --variable-columns temp humidity pressure
```

