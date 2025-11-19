import os
import sys
from typing import Dict, Tuple
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add parent 'main model' to sys.path to import data/eval utilities
_main_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'main model'))
if _main_model_path not in sys.path:
    sys.path.insert(0, _main_model_path)

from evaluation_utils import compute_forecasting_metrics, denormalize_forecasts


def set_seed(seed: int) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_dataloaders(
    loader,
    input_len: int,
    pred_len: int,
    batch_size: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    stride: int = 1,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Create train/val/test DataLoaders from the provided loader.
    """
    splits = loader.create_train_val_test_splits(
        input_length=input_len,
        prediction_length=pred_len,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        stride=stride,
        as_torch=True,
    )

    (x_train, y_train) = splits["train"]
    (x_val, y_val) = splits["val"]
    (x_test, y_test) = splits["test"]

    train_dl = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True, drop_last=False)
    val_dl = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False, drop_last=False)
    test_dl = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False, drop_last=False)
    return train_dl, val_dl, test_dl, splits


def batch_predict(model: nn.Module, xb: torch.Tensor) -> torch.Tensor:
    """
    Forward pass wrapper.
    xb: (B, T_in, M) -> returns forecasts (B, M, H)
    """
    out = model(xb)
    return out["forecasts"]


def evaluate_loop(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    norm_stats: Dict,
    denorm: bool = True,
) -> Dict[str, float]:
    """
    Run evaluation over a dataloader, return overall metrics dict.
    """
    model.eval()
    preds_all, targs_all = [], []
    with torch.no_grad():
        for xb, yb in data_loader:
            xb = xb.to(device)  # (B, T_in, M)
            yb = yb.to(device)  # (B, H, M)
            pred = batch_predict(model, xb)  # (B, M, H)
            pred = pred.permute(0, 2, 1).contiguous()  # (B, H, M)
            preds_all.append(pred)
            targs_all.append(yb)

    preds = torch.cat(preds_all, dim=0)  # (N, H, M)
    targs = torch.cat(targs_all, dim=0)  # (N, H, M)

    if denorm and norm_stats and norm_stats.get("method") != "none":
        preds_mh = preds.permute(0, 2, 1).contiguous()
        targs_mh = targs.permute(0, 2, 1).contiguous()
        preds_dn_mh = denormalize_forecasts(preds_mh, norm_stats)
        targs_dn_mh = denormalize_forecasts(targs_mh, norm_stats)
        preds_dn = preds_dn_mh.permute(0, 2, 1).contiguous()
        targs_dn = targs_dn_mh.permute(0, 2, 1).contiguous()
        metrics = compute_forecasting_metrics(preds_dn, targs_dn)
    else:
        metrics = compute_forecasting_metrics(preds, targs)

    return {k: float(v) for k, v in metrics.items()}


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, loss_fn, device) -> float:
    model.train()
    running = 0.0
    n = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)  # (B, H, M)
        optimizer.zero_grad()
        forecasts = batch_predict(model, xb)  # (B, M, H)
        forecasts = forecasts.permute(0, 2, 1).contiguous()  # (B, H, M)
        loss = loss_fn(forecasts, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running += float(loss) * xb.size(0)
        n += xb.size(0)
    return running / max(n, 1)





