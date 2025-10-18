import pandas as pd

def check_beta_values(df):
    """
    Verify that all methylation values are valid beta values (between 0 and 1).
    """
    if not ((df >= 0).all().all() and (df <= 1).all().all()):
        raise ValueError("Input methylation matrix contains non-beta values (outside [0,1]).")

def check_missing(df):
    """
    Check if there are missing values in the dataframe.
    """
    if df.isnull().values.any():
        raise ValueError("Input matrix contains missing values. Please impute or remove them before analysis.")

def align_cpg_sites(methylation, cpg_path):
    """
    Align CpG sites between input data and model requirements.

    If certain CpG sites are missing, they are filled with 0.5 (neutral methylation value).
    """
    cpg = pd.read_csv(cpg_path)["cpg"].tolist()
    missing_cols = [m for m in cpg if m not in methylation.columns]
    methylation = pd.concat([methylation, pd.DataFrame(0.5, index=methylation.index,
                                                       columns=missing_cols)], axis=1).loc[:, cpg].copy()
    return methylation
