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
# Use different config file
python train_extended_model.py --config my_config.json

# Override data path
python train_extended_model.py --data-path different_data.csv

# Override variable columns
python train_extended_model.py --variable-columns temp humidity pressure
```


### That's it! The model automatically uses your dataset configuration.

