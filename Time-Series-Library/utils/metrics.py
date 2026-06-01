import numpy as np


def _finite_pair(pred, true):
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    mask = np.isfinite(pred) & np.isfinite(true)
    if not np.any(mask):
        return np.array([0.0], dtype=np.float64), np.array([0.0], dtype=np.float64)
    return pred[mask], true[mask]


def RSE(pred, true):
    pred, true = _finite_pair(pred, true)
    denom = np.sqrt(np.sum((true - true.mean()) ** 2))
    return np.sqrt(np.sum((true - pred) ** 2)) / max(denom, 1e-12)


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    pred, true = _finite_pair(pred, true)
    return np.mean(np.abs(true - pred))


def MSE(pred, true):
    pred, true = _finite_pair(pred, true)
    return np.mean((true - pred) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    pred, true = _finite_pair(pred, true)
    denom = np.maximum(np.abs(true), 1e-6)
    return np.mean(np.abs((true - pred) / denom))


def MSPE(pred, true):
    pred, true = _finite_pair(pred, true)
    denom = np.maximum(np.abs(true), 1e-6)
    return np.mean(np.square((true - pred) / denom))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe
