# PerSEClock: A multi-organization epigenetic age prediction based on a channel attention perceptron networks

[**PerSEClock**](https://doi.org/10.3389/fgene.2024.1393856) was trained on 24,516 CpG loci that can utilize the samples from all types of methylation identification platforms and tested on 15 independent datasets. 
PerSEClock demonstrated the ability to assign varying weights to different CpG loci. 
This feature allows the model to enhance the weight of age-related loci while reducing the weight of irrelevant loci. 
The method is free to use for academics at www.dnamclock.com/#/original.  

---

## 🚀 1. Installation
Make sure all dependencies listed in requirements.txt are satisfied.
You can install them manually via:
```bash
pip install -r requirements.txt
```

You can clone this repository using the following command:
```bash
git clone https://github.com/NENUBioCompute/PerSEClock.git
```

## 📂 2. Project Structure
```bash
PerSEClock/
│
├── perseclock/
│   ├── __init__.py        # Package initialization
│   ├── dataset.py         # Data loading and preprocessing
│   ├── model.py           # Core model definitions
│   ├── evaluate.py        # Predicts biological age using a trained model.
│   ├── utils.py           # Helper functions and configuration tools
│   ├── visualize.py       # Visualization and plotting tools
│   └── main.py            # Main pipeline entry point
│
├── example_data/          # Example input datasets
│   ├── example1_beta.csv.gz
│   ├── example1_pheno.csv"
│
├── docs/                  # Documentation
│   └── tutorial.md
│
├── requirements.txt       # Package dependencies
└── setup.py               # Setup script for installation
```

## 🧩 3. Quick Start Example
After installation, you can use the following example to test the pipeline.
```python
from perseclock.main import run_perseclock

# Example usage
preds = run_perseclock(
    methylation_path="./example_data/example1_beta.csv.gz", # Input methylation beta-value matrix
    pheno_path="./example_data/example1_pheno.csv", # Input phenotype data with true ages
    model_path="./perseclock/PerSE_model.pt",  # Pre-trained model path
    device="cpu", # Use CPU for inference
    plot=True, # Enable plotting
    save_path="./results/PerSEClock_Predicted_Ages.csv" # Path to save predictions
)
```
The complete PerSEClock age prediction pipeline includes：
```python
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from perseclock.dataset import MethylationDataset
from perseclock.model import MLP
from perseclock.evaluate import predict_age
from perseclock.utils import check_beta_values, check_missing, align_cpg_sites
from perseclock.visualize import plot_predicted_vs_true


# Step 1: Load data
methylation = pd.read_csv(methylation_path, index_col=0) # CpG sites as rows, samples as columns
# Transpose methylation data
methylation = methylation.transpose()  # Samples as rows, CpG sites as columns
# Data quality checks
check_missing(methylation)  # Ensure no missing values
check_beta_values(methylation)  # Ensure beta values are valid
# Align CpG sites
methylation = align_cpg_sites(methylation, './perseclock/24516cpg.csv')

# Step 2: Prepare DataLoader
X = methylation.values.astype("float32")
# Create Dataset and DataLoader
dataset = MethylationDataset(X)
loader = DataLoader(dataset, batch_size=len(X), shuffle=False)

# Step 3: Load model
model = MLP(in_features=X.shape[1])
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)

# Step 4: Predict
pred_age = predict_age(model, loader, device)

# Step 5: Visualization (optional)
if plot:
    plot_predicted_vs_true(pred_age, true_age, save_path='predicted_vs_true.png')

# Step 6: Save predictions
if save_path:
    output_df = pd.DataFrame({'Predicted_Age': pred_age})
    output_df.to_csv(save_path, index=False)
```

## 📘 4. Example Data Format
Input datasets should be in .csv format. 
The methylation expression matrix and the phenotype file containing chronic age are stored separately in the _beta.csv and _pheno.csv files.

**Input methylation beta-value matrix** with the following structure:
|SampleID	|S1	|S2	|...	|Sn |
| - | - | - | - | - |
|CpG_1 |0.75	|0.62	|...	|0.81|
|CpG_2 |0.62	|0.86	|...	|0.90|
|... |...	|...	|...	|...|
|CpG_n |0.88	|0.59	|...	|0.79|


Each row represents a methylation site or feature
Each column represents a sample

**Input target variable (ChronAge)** should be numeric in _pheno.csv

## 📊 4. Example Results
Use the example file as input, run the [test_perseclock.py](https://github.com/NENUBioCompute/PerSEClock/blob/main/test_perseclock.py) , 
and obtain the output shown below:
```bash
[INFO] Successfully predicted 78 samples.
[INFO] MAE: 1.23, R²: 0.87, RMSE: 1.45
[INFO] Predictions saved to './results/PerSEClock_Predicted_Ages.csv'.
```

