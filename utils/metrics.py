import numpy as np


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def metric(pred, true):
    """Return (mae, mse, rmse) in the standardized evaluation space."""
    return MAE(pred, true), MSE(pred, true), RMSE(pred, true)
