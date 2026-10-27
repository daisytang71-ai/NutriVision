"""Metrics in physical units; undefined R² is JSON null, never invented."""
import numpy as np
from . import TARGETS, UNITS


def regression_metrics(y, pred):
    y, pred = np.asarray(y, dtype=np.float64), np.asarray(pred, dtype=np.float64)
    if y.shape != pred.shape or y.ndim != 2 or y.shape[1] != 4 or len(y) == 0:
        raise ValueError("Expected matching non-empty (N,4) arrays")
    if not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ValueError("Non-finite targets/predictions")
    result = {}
    for i, (name, unit) in enumerate(zip(TARGETS, UNITS)):
        err = pred[:, i] - y[:, i]
        sst = np.square(y[:, i] - y[:, i].mean()).sum()
        r2 = float(1 - np.square(err).sum() / sst) if len(y) >= 2 and sst > 0 else None
        result[name] = {"MAE": float(np.abs(err).mean()),
                        "RMSE": float(np.sqrt(np.square(err).mean())), "R2": r2, "unit": unit}
    return result


def aggregate_dishes(ids, y, pred):
    order = sorted(set(ids))
    y, pred, ids = np.asarray(y), np.asarray(pred), np.asarray(ids)
    truth, prediction = [], []
    for dish in order:
        mask = ids == dish
        if not np.allclose(y[mask], y[mask][0], atol=1e-3, rtol=1e-5):
            raise ValueError(f"Inconsistent per-image dish labels: {dish}")
        truth.append(y[mask][0])
        prediction.append(pred[mask].mean(0))
    return order, np.asarray(truth), np.asarray(prediction)
