from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _r2_numpy(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    residual = np.mean((target - prediction) ** 2, axis=0)
    total = np.var(target, axis=0)
    return np.nan_to_num(1 - residual / total)


def _ridge_numpy(features: np.ndarray, target: np.ndarray, regularization: float) -> np.ndarray:
    features = np.asarray(features)
    return np.linalg.solve(
        features.T @ features + regularization * np.eye(features.shape[1]),
        features.T @ target,
    )


def fit_cpu_ridge(
    features: np.ndarray,
    target: np.ndarray,
    *,
    lambdas: Sequence[float],
    splits: int = 10,
) -> np.ndarray:
    """Run ordered K-fold selection independently for every target."""
    from sklearn.model_selection import KFold

    costs = np.zeros((len(lambdas), target.shape[1]))
    for train_indices, validation_indices in KFold(n_splits=splits).split(target):
        for index, regularization in enumerate(lambdas):
            weights = _ridge_numpy(features[train_indices], target[train_indices], regularization)
            costs[index] += 1 - _r2_numpy(
                features[validation_indices] @ weights, target[validation_indices]
            )
    selected = np.argmin(costs, axis=0)
    weights = np.zeros((features.shape[1], target.shape[1]))
    for index, regularization in enumerate(lambdas):
        mask = selected == index
        if np.any(mask):
            weights[:, mask] = _ridge_numpy(features, target[:, mask], regularization)
    return weights


def fit_torch_ridge(features, target, *, lambdas: Sequence[float], splits: int = 10):
    """Select one GPU ridge penalty by summed target error."""
    import torch
    from sklearn.model_selection import KFold

    def solve(x, y, regularization):
        identity = regularization * torch.eye(x.shape[1], device=x.device)
        return torch.linalg.lstsq(x.T @ x + identity, x.T @ y).solution

    def r2(prediction, actual):
        residual = torch.mean((prediction - actual) ** 2, dim=0)
        total = torch.var(actual, dim=0)
        return torch.nan_to_num(1 - residual / total)

    errors = torch.zeros(len(lambdas), device=features.device)
    for train_indices, validation_indices in KFold(n_splits=splits).split(target.cpu()):
        train_x, train_y = features[train_indices], target[train_indices]
        validation_x, validation_y = features[validation_indices], target[validation_indices]
        for index, regularization in enumerate(lambdas):
            weights = solve(train_x, train_y, regularization)
            errors[index] += torch.sum(1 - r2(validation_x @ weights, validation_y))
    selected = int(torch.argmin(errors).item())
    return solve(features, target, lambdas[selected])
