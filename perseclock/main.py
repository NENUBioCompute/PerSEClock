import torch
import pandas as pd
from torch.utils.data import DataLoader
from .dataset import MethylationDataset
from .model import MLP
from .evaluate import predict_age
from .utils import check_beta_values, check_missing, align_cpg_sites
from .visualize import plot_predicted_vs_true
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import numpy as np

def run_perseclock(methylation_path,  model_path, pheno_path=None, device="cpu", plot=True, save_path=None):
    """
    Run the complete PerSEClock age prediction pipeline.

    Parameters
    ----------
    methylation_path : str
        Path to methylation beta-value matrix (.csv).
    model_path : str
        Path to pre-trained PerSEClock model weights (.pt).
    pheno_path : str
        Path to methylation pheno information (.csv).
    device : str
        Device for inference ('cpu' or 'cuda').
    plot : bool
        Whether to visualize predicted vs true ages.
    save_path : str
        Path to saved predicted results (.csv).
    Returns
    -------
    list
        Predicted biological ages.
    """
    # Step 1: Load data
    methylation = pd.read_csv(methylation_path, index_col=0) # Rows: CpG sites, Columns: Samples
    pheno = pd.read_csv(pheno_path) if pheno_path else pd.DataFrame() # Load pheno if provided
    # Filter pheno to match methylation samples
    pheno = pheno[pheno['ID'].isin(methylation.columns)].reset_index(drop=True) if not pheno.empty else pheno
    # Transpose methylation data
    methylation = methylation.transpose() # Rows: Samples, Columns: CpG sites
    # Data checks and alignment
    check_missing(methylation)
    check_beta_values(methylation)
    # Align CpG sites
    methylation = align_cpg_sites(methylation, './perseclock/24516cpg.csv')

    # Step 2: Prepare DataLoader
    X = methylation.values.astype("float32")
    # Process true ages if pheno is provided
    if not pheno.empty:
        if pheno['Age_unit'].values[0] == 'Month':
            true_age = pheno['Age'].astype('float64')/12
        elif pheno['Age_unit'].values[0] == 'Year':
            true_age = pheno['Age'].astype('float64').values
        elif pheno['Age_unit'].values[0] == 'Day':
            true_age = pheno['Age'].astype('float64')/365
        elif pheno['Age_unit'].values[0] == 'Week':
            true_age = pheno['Age'].astype('float64')/52
        else:
            print('Age_unit Error!')
    else:
        true_age = None

    # Create Dataset and DataLoader
    dataset = MethylationDataset(X)
    loader = DataLoader(dataset, batch_size=len(X), shuffle=False)

    # Step 3: Load model
    model = MLP(in_features=X.shape[1])
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)

    # Step 4: Predict
    pred_age = predict_age(model, loader, device)
    print(f"[INFO] Successfully predicted {len(pred_age)} samples.")

    # Step 5: Visualization (optional)
    if plot:
        # Here we simulate true ages for demo purposes
        assert true_age is not None, "True ages are required for plotting."
        plot_predicted_vs_true(pred_age, true_age, save_path='./results/predicted_vs_true.png')
        # Calculate and print metrics
        mae = mean_absolute_error(true_age, pred_age)
        r2 = r2_score(true_age, pred_age)
        rmse = np.sqrt(mean_squared_error(true_age, pred_age))
        print(f"[INFO] MAE: {mae:.2f}, R²: {r2:.2f}, RMSE: {rmse:.2f}")

    # Step 6: Save predictions
    if save_path:
        output_df = pd.DataFrame({'Predicted_Age': pred_age})
        output_df.to_csv(save_path, index=False)
        print(f"[INFO] Predictions saved to '{save_path}'.")

    return pred_age
