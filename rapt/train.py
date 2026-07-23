"""Training loop for RAPT (AdamW + OneCycleLR + AMP, denoising inputs)."""

from __future__ import annotations

import copy
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .config import RaptConfig
from .data import NormStats, SequenceData, make_windows
from .model import RaptModel, nll_loss


def train_rapt(
    cfg: RaptConfig,
    train_data: SequenceData,
    val_data: SequenceData | None = None,
    device: str | None = None,
    stats: NormStats | None = None,
    log_fn=print,
) -> tuple[RaptModel, NormStats, dict]:
    """Train a RAPT model on nominal sequences.

    Returns ``(model, norm_stats, history)`` where ``model`` carries the best
    validation weights (or final weights when no validation set is given) and
    ``history`` holds per-epoch train/val NLL.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    if stats is None:
        stats = NormStats.from_data(train_data)
    x_tr, y_tr = make_windows(train_data, cfg.seq_len, stats, cfg.train_dynamics)
    loaders = {
        "train": DataLoader(
            TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr)),
            batch_size=cfg.batch_size,
            shuffle=True,
            drop_last=False,
        )
    }
    if val_data is not None and len(val_data):
        x_va, y_va = make_windows(val_data, cfg.seq_len, stats, cfg.train_dynamics)
        loaders["val"] = DataLoader(
            TensorDataset(torch.from_numpy(x_va), torch.from_numpy(y_va)),
            batch_size=cfg.batch_size,
        )

    model = RaptModel(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.lr, epochs=cfg.num_epochs, steps_per_epoch=len(loaders["train"])
    )
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    history: dict = {"train_nll": [], "val_nll": []}
    best_val, best_state = float("inf"), None
    n_params = sum(p.numel() for p in model.parameters())
    log_fn(
        f"Training RAPT on {device}: {len(x_tr)} windows of {cfg.seq_len} steps, "
        f"input_dim={cfg.input_dim}, obs_dim={cfg.obs_dim}, {n_params / 1e6:.2f}M params, "
        f"target={'forward-dynamics' if cfg.train_dynamics else 'reconstruction'}"
    )

    for epoch in range(cfg.num_epochs):
        t0 = time.time()
        model.train()
        train_losses = []
        for x, y in loaders["train"]:
            x, y = x.to(device), y.to(device)
            x = x + torch.randn_like(x) * cfg.noise_scale
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=device == "cuda"):
                mu, log_var, _ = model(x, mask=model.cfg.reconstruction_type == "masked")
                loss = nll_loss(mu, log_var, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            train_losses.append(loss.item())
        history["train_nll"].append(float(np.mean(train_losses)))

        msg = (
            f"epoch {epoch + 1:3d}/{cfg.num_epochs} | "
            f"train NLL {history['train_nll'][-1]:+.4f}"
        )
        if "val" in loaders:
            val_nll = evaluate_nll(model, loaders["val"], device)
            history["val_nll"].append(val_nll)
            msg += f" | val NLL {val_nll:+.4f}"
            if val_nll < best_val:
                best_val = val_nll
                best_state = copy.deepcopy(model.state_dict())
                msg += " *"
        log_fn(msg + f" | {time.time() - t0:.1f}s")

    if best_state is not None:
        model.load_state_dict(best_state)
        log_fn(f"Restored best validation weights (val NLL {best_val:+.4f}).")
    model.eval()
    return model, stats, history


@torch.no_grad()
def evaluate_nll(model: RaptModel, loader, device: str) -> float:
    model.eval()
    losses = []
    for x, y in loader:
        mu, log_var, _ = model(x.to(device))
        losses.append(nll_loss(mu, log_var, y.to(device)).item())
    return float(np.mean(losses))
