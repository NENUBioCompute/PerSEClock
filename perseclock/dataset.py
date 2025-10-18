import numpy as np
import torch
from torch.utils.data import Dataset

class MethylationDataset(Dataset):
    """
    Custom PyTorch dataset for DNA methylation data.

    Parameters
    ----------
    feature_array : np.ndarray
        Input methylation beta value matrix (samples x CpGs).
    dtype : np.dtype
        Data type for conversion (default: np.float32).
    """
    def __init__(self, feature_array: np.ndarray, dtype=np.float32):
        if not isinstance(feature_array, np.ndarray):
            raise ValueError("`feature_array` must be a numpy.ndarray.")
        self.features = feature_array.astype(dtype)

    def __getitem__(self, idx):
        """Return a single sample by index."""
        return self.features[idx]
    
    def __len__(self):
        """Return total number of samples."""
        return self.features.shape[0]
