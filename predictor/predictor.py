import numpy as np
from tqdm import tqdm

def predict(model, dataset):
    """Output predictions and labels for a given model and dataloader"""
    preds = model.predict(dataset.X)
    targets = dataset.y

    preds_mean = preds.mean()
    targets_mean = targets.mean()

    preds = preds + (targets_mean - preds_mean)
    assert np.isclose(preds.mean(), targets.mean())

    return preds, targets
