import torch

def predict_age(model, dataloader, device):
    """
    Predicts biological age using a trained model.

    Parameters
    ----------
    model : torch.nn.Module
        Trained MLP model.
    dataloader : torch.utils.data.DataLoader
        Test data loader.
    device : str
        Device to run inference ("cpu" or "cuda").

    Returns
    -------
    list
        Predicted biological ages.
    """
    model.eval()
    preds = []
    with torch.no_grad():
        for features in dataloader:
            features = features.to(device)
            pred = model(features)
            preds.extend(pred.cpu().numpy().tolist())
    return preds
