from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from torch import nn


SEED = 20260918
MODEL_NAMES = ("model0_snapshot", "model1_elastic_net", "model2_xgboost", "model3_grud")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(2)


def fit_tabular_preprocessor(x: np.ndarray, scale: bool) -> dict[str, Any]:
    with np.errstate(all="ignore"):
        median = np.nanmedian(x, axis=0)
    median = np.where(np.isfinite(median), median, 0.0)
    filled = np.where(np.isfinite(x), x, median)
    mean = filled.mean(axis=0) if scale else np.zeros(x.shape[1])
    std = filled.std(axis=0) if scale else np.ones(x.shape[1])
    std = np.where(std > 1e-8, std, 1.0)
    return {"median": median.tolist(), "mean": mean.tolist(), "std": std.tolist(), "scale": bool(scale)}


def apply_tabular_preprocessor(x: np.ndarray, prep: dict[str, Any]) -> np.ndarray:
    median = np.asarray(prep["median"], dtype=float)
    mean = np.asarray(prep["mean"], dtype=float)
    std = np.asarray(prep["std"], dtype=float)
    filled = np.where(np.isfinite(x), x, median)
    return ((filled - mean) / std).astype(np.float32)


def fit_sequence_preprocessor(values: np.ndarray, age: np.ndarray) -> dict[str, Any]:
    with np.errstate(all="ignore"):
        mean = np.nanmean(values, axis=(0, 1))
        std = np.nanstd(values, axis=(0, 1))
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where(std > 1e-8, std, 1.0)
    age_mean = float(np.mean(age))
    age_std = float(np.std(age))
    if age_std <= 1e-8:
        age_std = 1.0
    return {"value_mean": mean.tolist(), "value_std": std.tolist(),
            "age_mean": age_mean, "age_std": age_std}


def apply_sequence_preprocessor(values: np.ndarray, mask: np.ndarray, age: np.ndarray,
                                sex: np.ndarray, prep: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(prep["value_mean"], dtype=np.float32)
    std = np.asarray(prep["value_std"], dtype=np.float32)
    normalized = (values - mean[None, None, :]) / std[None, None, :]
    normalized = np.where(mask > 0, normalized, 0.0).astype(np.float32)
    static = np.column_stack([(age - prep["age_mean"]) / prep["age_std"], sex]).astype(np.float32)
    return normalized, static


def _irls(y: np.ndarray, x: np.ndarray, offset: np.ndarray | None = None, max_iter: int = 50) -> np.ndarray:
    beta = np.zeros(x.shape[1], dtype=float)
    if offset is None:
        offset = np.zeros(len(y), dtype=float)
    for _ in range(max_iter):
        eta = np.clip(offset + x @ beta, -30, 30)
        p = 1.0 / (1.0 + np.exp(-eta))
        w = np.maximum(p * (1.0 - p), 1e-8)
        gradient = x.T @ (y - p)
        hessian = (x.T * w) @ x
        step = np.linalg.pinv(hessian) @ gradient
        beta_new = beta + step
        if np.max(np.abs(step)) < 1e-9:
            beta = beta_new
            break
        beta = beta_new
    return beta


def calibration_metrics(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, dtype=float)
    lp = np.log(p / (1.0 - p))
    intercept = float(_irls(y, np.ones((len(y), 1)), offset=lp)[0])
    joint = _irls(y, np.column_stack([np.ones(len(y)), lp]))
    return intercept, float(joint[1])


def metric_values(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    intercept, slope = calibration_metrics(y, p)
    return {
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "prevalence": float(np.mean(y)),
    }


def bootstrap_performance(y: np.ndarray, predictions: dict[str, np.ndarray], replicates: int = 2000,
                          seed: int = SEED, compute_differences: bool = True, reference: str | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    point = {name: metric_values(y, p) for name, p in predictions.items()}
    metrics = ("auroc", "auprc", "brier", "log_loss", "calibration_intercept", "calibration_slope")
    boot = {name: {metric: [] for metric in metrics} for name in predictions}
    difference_rows: list[dict[str, Any]] = []
    valid_reps = 0
    indexes_used: list[np.ndarray] = []
    for _ in range(replicates):
        idx = rng.integers(0, len(y), len(y))
        if np.unique(y[idx]).size < 2:
            continue
        indexes_used.append(idx)
        valid_reps += 1
        for name, p in predictions.items():
            values = metric_values(y[idx], p[idx])
            for metric in metrics:
                boot[name][metric].append(values[metric])
    rows: list[dict[str, Any]] = []
    for name in predictions:
        for metric in metrics:
            vals = np.asarray(boot[name][metric], dtype=float)
            rows.append({"model": name, "metric": metric, "estimate": point[name][metric],
                         "ci_low": float(np.quantile(vals, .025)), "ci_high": float(np.quantile(vals, .975)),
                         "bootstrap_replicates": valid_reps})
    if not compute_differences:
        return rows, difference_rows
    reference = reference or ("model3_grud" if "model3_grud" in predictions else list(predictions)[0])
    for comparator in predictions:
        if comparator == reference:
            continue
        for metric in ("auroc", "auprc", "brier", "log_loss"):
            vals = np.asarray(boot[reference][metric]) - np.asarray(boot[comparator][metric])
            difference_rows.append({"reference_model": reference, "comparator_model": comparator,
                                    "metric": metric, "estimate": point[reference][metric] - point[comparator][metric],
                                    "ci_low": float(np.quantile(vals, .025)), "ci_high": float(np.quantile(vals, .975)),
                                    "bootstrap_replicates": len(vals), "paired": True})
    return rows, difference_rows


class CompactGRUD(nn.Module):
    def __init__(self, n_features: int, hidden_size: int, dropout: float):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.decay_x_weight = nn.Parameter(torch.zeros(n_features))
        self.decay_x_bias = nn.Parameter(torch.zeros(n_features))
        self.decay_h = nn.Linear(n_features, hidden_size)
        self.gru_cell = nn.GRUCell(n_features * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size + 2, 1)

    def forward(self, values: torch.Tensor, mask: torch.Tensor,
                delta: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        batch = values.shape[0]
        hidden = torch.zeros(batch, self.hidden_size, device=values.device, dtype=values.dtype)
        last = torch.zeros(batch, self.n_features, device=values.device, dtype=values.dtype)
        for t in range(values.shape[1]):
            d = delta[:, t, :]
            gamma_x = torch.exp(-torch.relu(d * self.decay_x_weight + self.decay_x_bias))
            gamma_h = torch.exp(-torch.relu(self.decay_h(d)))
            hidden = hidden * gamma_h
            current_mask = mask[:, t, :]
            current = values[:, t, :]
            imputed = current_mask * current + (1.0 - current_mask) * gamma_x * last
            hidden = self.gru_cell(torch.cat([imputed, current_mask], dim=1), hidden)
            last = current_mask * current + (1.0 - current_mask) * last
        logits = self.output(torch.cat([self.dropout(hidden), static], dim=1)).squeeze(1)
        return logits


def parameter_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


@dataclass
class GrudFit:
    state_dict: dict[str, torch.Tensor]
    best_epoch: int
    validation_log_loss: float
    validation_brier: float
    parameter_count: int


def fit_grud(x_train: np.ndarray, m_train: np.ndarray, d_train: np.ndarray, static_train: np.ndarray,
             y_train: np.ndarray, x_valid: np.ndarray, m_valid: np.ndarray, d_valid: np.ndarray,
             static_valid: np.ndarray, y_valid: np.ndarray, hidden_size: int, dropout: float,
             weight_decay: float, seed: int, weighted: bool = True, max_epochs: int = 100,
             patience: int = 10) -> GrudFit:
    set_seed(seed)
    model = CompactGRUD(x_train.shape[2], hidden_size, dropout)
    count = parameter_count(model)
    if count >= 20000:
        raise RuntimeError(f"GRU-D parameter count {count} violates <20000")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=weight_decay)
    positives = float(y_train.sum())
    pos_weight = (len(y_train) - positives) / positives if weighted and positives > 0 else 1.0
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32))
    train_tensors = [torch.as_tensor(v, dtype=torch.float32) for v in
                     (x_train, m_train, d_train, static_train, y_train)]
    valid_tensors = [torch.as_tensor(v, dtype=torch.float32) for v in
                     (x_valid, m_valid, d_valid, static_valid)]
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = math.inf
    best_brier = math.inf
    best_epoch = 0
    stalled = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(*train_tensors[:4])
        loss = criterion(logits, train_tensors[4])
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            pred = torch.sigmoid(model(*valid_tensors)).cpu().numpy()
        valid_loss = float(log_loss(y_valid, pred, labels=[0, 1]))
        valid_brier = float(brier_score_loss(y_valid, pred))
        improved = valid_loss < best_loss - 1e-6 or (abs(valid_loss - best_loss) <= 1e-6 and valid_brier < best_brier)
        if improved:
            best_loss, best_brier, best_epoch = valid_loss, valid_brier, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stalled = 0
        else:
            stalled += 1
            if stalled >= patience:
                break
    if best_state is None:
        raise RuntimeError("GRU-D training failed to produce a checkpoint")
    return GrudFit(best_state, best_epoch, best_loss, best_brier, count)


def fit_grud_fixed_epochs(x: np.ndarray, mask: np.ndarray, delta: np.ndarray, static: np.ndarray,
                          y: np.ndarray, hidden_size: int, dropout: float, weight_decay: float,
                          epochs: int, seed: int, weighted: bool = True) -> CompactGRUD:
    set_seed(seed)
    model = CompactGRUD(x.shape[2], hidden_size, dropout)
    if parameter_count(model) >= 20000:
        raise RuntimeError("GRU-D parameter limit violated")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=weight_decay)
    positives = float(y.sum())
    pos_weight = (len(y) - positives) / positives if weighted and positives > 0 else 1.0
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32))
    tensors = [torch.as_tensor(v, dtype=torch.float32) for v in (x, mask, delta, static, y)]
    for _ in range(max(1, int(epochs))):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(*tensors[:4]), tensors[4])
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def predict_grud(model: CompactGRUD, x: np.ndarray, mask: np.ndarray, delta: np.ndarray,
                 static: np.ndarray, batch_size: int = 512) -> np.ndarray:
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            stop = min(start + batch_size, len(x))
            tensors = [torch.as_tensor(v[start:stop], dtype=torch.float32) for v in (x, mask, delta, static)]
            predictions.append(torch.sigmoid(model(*tensors)).cpu().numpy())
    return np.concatenate(predictions).astype(float)


def save_json(obj: Any, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
