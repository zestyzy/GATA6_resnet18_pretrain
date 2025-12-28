# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import os
import random
import math
import sys
from contextlib import contextmanager
from datetime import datetime
from typing import Tuple, List, Dict, Any, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve,
    f1_score, matthews_corrcoef, balanced_accuracy_score
)
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.manifold import TSNE  # [New] For visualization
from sklearn.decomposition import PCA # [New] Optional backup
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import matplotlib
matplotlib.use("Agg")  # Prevent backend errors in headless environments
import matplotlib.pyplot as plt
import pandas as pd
import warnings

# Ensure dataset.py and resnet3d.py are in the same directory or Python path
from .dataset import NiftiDataset, LightAugment3D, ChannelNormalize, Compose3D
from .resnet3d import (
    generate_resnet18, generate_resnet34, generate_resnet50,
    bn_factory, gn_factory, in_factory,
    load_medicalnet_weights, load_pretrained_2d_weights_to_3d
)

# =============================================================================
# [NEW] Tee logger: duplicate stdout/stderr to file (per-repeat train.log)
# =============================================================================
class _TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
            except Exception:
                # Don't break training if a stream fails
                pass

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self):
        # Helps tqdm detect terminal capability
        try:
            return any(getattr(s, "isatty", lambda: False)() for s in self.streams)
        except Exception:
            return False

@contextmanager
def tee_stdout_stderr(log_path: str, mode: str = "w", encoding: str = "utf-8"):
    """
    Context manager that tees BOTH stdout and stderr to a log file,
    while still keeping original console output.
    This captures tqdm (stderr) + print (stdout).
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    orig_out, orig_err = sys.stdout, sys.stderr
    f = open(log_path, mode=mode, encoding=encoding, buffering=1)  # line-buffered
    try:
        sys.stdout = _TeeStream(orig_out, f)
        sys.stderr = _TeeStream(orig_err, f)
        yield
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        sys.stdout, sys.stderr = orig_out, orig_err
        try:
            f.flush()
        except Exception:
            pass
        f.close()

# ----------------------- Subset with on-the-fly transform -----------------------
class SubsetWithTransform(Dataset):
    def __init__(self, dataset: Dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = list(indices)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.dataset[self.indices[i]]  # (C,D,H,W), label
        if self.transform is not None:
            x = self.transform(x)
        return x, y

# ----------------------- Reproducibility -----------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ----------------------- Metrics -----------------------
def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    N = len(probs)
    for i in range(n_bins):
        lower, upper = bin_boundaries[i], bin_boundaries[i + 1]
        if i < n_bins - 1:
            mask = (probs >= lower) & (probs < upper)
        else:
            mask = (probs >= lower) & (probs <= upper)
        if mask.sum() > 0:
            bin_acc = labels[mask].mean()
            bin_conf = probs[mask].mean()
            ece += (mask.sum() / N) * abs(bin_conf - bin_acc)
    return float(ece)

def pick_threshold_by_youden(labels: np.ndarray, probs_pos: np.ndarray) -> float:
    fpr, tpr, thr = roc_curve(labels, probs_pos)
    if len(thr) == 0:
        return 0.5
    youden = tpr - fpr
    i = int(np.argmax(youden))
    return float(thr[i])

def summarize_at_threshold(labels: np.ndarray, probs_pos: np.ndarray, thr: float) -> Dict[str, float]:
    preds = (probs_pos >= thr).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    sens = tp / max(1, tp + fn)
    spec = tn / max(1, tn + fp)
    ppv = tp / max(1, tp + fp)
    npv = tn / max(1, tn + fn)
    mcc = matthews_corrcoef(labels, preds) if (tp + tn + fp + fn) > 0 else 0.0
    bal = balanced_accuracy_score(labels, preds) if (tp + tn + fp + fn) > 0 else 0.0
    f1 = f1_score(labels, preds) if (tp + fp) > 0 else 0.0
    return dict(thr=float(thr), sens=float(sens), spec=float(spec),
                ppv=float(ppv), npv=float(npv), mcc=float(mcc),
                bal_acc=float(bal), f1=float(f1))

# ----------------------- Dataset statistics -----------------------
def compute_dataset_channel_stats(
    dataset: Dataset,
    channels: Sequence[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not channels:
        raise ValueError("channels must be a non-empty sequence")
    sums = torch.zeros(len(channels), dtype=torch.float64)
    sq_sums = torch.zeros(len(channels), dtype=torch.float64)
    counts = torch.zeros(len(channels), dtype=torch.float64)
    for idx in tqdm(range(len(dataset)), desc="Compute mean/std", leave=False):
        vol, _ = dataset[idx]
        if not torch.is_tensor(vol):
            vol = torch.as_tensor(vol)
        vol = vol.to(dtype=torch.float64)
        for i, ch in enumerate(channels):
            if ch >= vol.size(0):
                continue
            data = vol[ch]
            sums[i] += data.sum().item()
            sq_sums[i] += (data * data).sum().item()
            counts[i] += data.numel()
    means = sums / torch.clamp(counts, min=1.0)
    vars = sq_sums / torch.clamp(counts, min=1.0) - means ** 2
    stds = torch.sqrt(torch.clamp(vars, min=1e-12))
    return means.to(dtype=torch.float32), stds.to(dtype=torch.float32)

# ----------------------- BN freeze helpers -----------------------
def freeze_bn_running_stats(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()
            if m.weight is not None:
                m.weight.requires_grad_(True)
            if m.bias is not None:
                m.bias.requires_grad_(True)

def _set_bn_eval(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()

def _current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])

# ----------------------- Helpers for BCE -----------------------
def _pos_logit(outputs: torch.Tensor, num_classes: int) -> torch.Tensor:
    if num_classes == 1:
        if outputs.ndim == 2 and outputs.size(1) == 1:
            return outputs.squeeze(1)
        elif outputs.ndim == 1:
            return outputs
        else:
            raise ValueError(f"Unexpected output shape {tuple(outputs.shape)} for num_classes=1")
    return outputs[:, 1]

# ----------------------- [New] Visualization Logic -----------------------
class FeatureExtractorHook:
    """
    Hook to capture features from the layer before the final FC.
    Assuming ResNet structure: ... -> avgpool -> fc
    """
    def __init__(self):
        self.features = []

    def hook_fn(self, module, input, output):
        # output is usually (B, C, 1, 1, 1) or (B, C) depending on avgpool implementation
        # We flatten it to (B, C)
        self.features.append(output.flatten(start_dim=1).detach().cpu())

    def clear(self):
        self.features = []

    def get_features(self):
        if not self.features:
            return None
        return torch.cat(self.features, dim=0).numpy()

def visualize_features(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    outdir: str,
    epoch: int,
    tag: str = "val"
):
    """
    Extracts features, runs t-SNE, plots scatter plot.
    """
    model.eval()

    # Register Hook on avgpool
    # Adjust this if your ResNet definition names avgpool differently
    if hasattr(model, 'avgpool'):
        target_layer = model.avgpool
    elif hasattr(model, 'module') and hasattr(model.module, 'avgpool'): # For DataParallel
        target_layer = model.module.avgpool
    else:
        print("[Vis] Warning: Could not find 'avgpool' layer. Visualization skipped.")
        return

    hook = FeatureExtractorHook()
    handle = target_layer.register_forward_hook(hook.hook_fn)

    all_labels = []

    try:
        with torch.no_grad():
            for volumes, labels in tqdm(dataloader, desc=f"Vis-{tag}", leave=False):
                volumes = volumes.to(device)
                _ = model(volumes) # Forward pass triggers hook
                all_labels.extend(labels.numpy())

        feats = hook.get_features()
        labels_np = np.array(all_labels)

        if feats is None or len(feats) == 0:
            print("[Vis] No features captured.")
            return

        # t-SNE Calculation
        # Perplexity must be < n_samples. Default 30.
        n_samples = feats.shape[0]
        perp = min(30, max(5, n_samples // 3)) # auto-adjust perplexity for small val sets

        print(f"[Vis] Computing t-SNE on {n_samples} samples (dim={feats.shape[1]})...")
        tsne = TSNE(n_components=2, perplexity=perp, random_state=42, init='pca', learning_rate='auto')
        feats_embedded = tsne.fit_transform(feats)

        # Plotting
        plt.figure(figsize=(8, 6))

        # Color map: 0=Blue, 1=Red
        colors = ['blue' if l == 0 else 'red' for l in labels_np]

        plt.scatter(feats_embedded[:, 0], feats_embedded[:, 1], c=colors, alpha=0.6, edgecolors='k', s=60)

        # Create legend manually
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Class 0 (Neg)', markerfacecolor='blue', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='Class 1 (Pos)', markerfacecolor='red', markersize=10)
        ]
        plt.legend(handles=legend_elements, loc="upper right")

        plt.title(f"t-SNE of Features (Epoch {epoch}) [{tag}]")
        plt.xlabel("Dim 1")
        plt.ylabel("Dim 2")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        vis_path = os.path.join(outdir, f"tsne_{tag}_ep{epoch:03d}.png")
        plt.savefig(vis_path, dpi=150)
        plt.close()
        print(f"[Vis] Saved t-SNE plot to {vis_path}")

    except Exception as e:
        print(f"[Vis] Error during visualization: {e}")
    finally:
        handle.remove() # Clean up hook
        hook.clear()

# ----------------------- [New] Diagnostics -----------------------
def debug_visualize_batch(volumes: torch.Tensor, labels: torch.Tensor, outdir: str, epoch: int, stage: str = "train"):
    """
    [Diagnostic Tool] Sample and save model input tensors as images.
    """
    vis_dir = os.path.join(outdir, "debug_vis")
    os.makedirs(vis_dir, exist_ok=True)
    batch_size = volumes.shape[0]

    # Check first 2 samples
    for i in range(min(2, batch_size)):
        vol = volumes[i].detach().cpu().numpy()  # (C, D, H, W)
        label = labels[i].item()

        # Middle slice
        mid_z = vol.shape[1] // 2

        # Image Channel
        img_slice = vol[0, mid_z]

        n_channels = vol.shape[0]
        fig, ax = plt.subplots(1, n_channels, figsize=(5 * n_channels, 5))
        if n_channels == 1:
            ax = [ax]

        # Draw CT
        im0 = ax[0].imshow(img_slice, cmap='gray', origin='lower')
        ax[0].set_title(f"Ep{epoch} | BatchSample {i} | Lbl {label}\nCT (z={mid_z}) Range:[{img_slice.min():.1f}, {img_slice.max():.1f}]")
        plt.colorbar(im0, ax=ax[0])

        # Draw Mask (if exists)
        if n_channels > 1:
            msk_slice = vol[1, mid_z]
            im1 = ax[1].imshow(msk_slice, cmap='jet', interpolation='nearest', origin='lower')
            ax[1].set_title(f"Mask (z={mid_z}) Range:[{msk_slice.min():.1f}, {msk_slice.max():.1f}]")
            plt.colorbar(im1, ax=ax[1])

        plt.tight_layout()
        save_path = os.path.join(vis_dir, f"{stage}_ep{epoch:03d}_sample{i}.png")
        plt.savefig(save_path)
        plt.close()

# ----------------------- Train / Eval -----------------------
def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    num_classes: int,
    max_grad_norm: float | None = 0.0,
    dbg_prob_log: bool = False,
    freeze_bn=False,
    scaler: GradScaler | None = None,
    amp_dtype: torch.dtype = torch.float16,
    epoch: int = 0,
    outdir: str = "",
) -> float:
    model.train()
    if freeze_bn:
        _set_bn_eval(model)

    running_loss = 0.0
    n_samples = 0
    pbar = tqdm(dataloader, desc=f"Train(lr={_current_lr(optimizer):.2e})", leave=False)
    first_dbg_done = False

    for i, (volumes, labels) in enumerate(pbar):
        # --- Diagnostic Visualization ---
        if i == 0 and epoch > 0 and (epoch % 5 == 0 or epoch == 1):
            debug_visualize_batch(volumes, labels, outdir, epoch, "train")

        volumes = volumes.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float()

        optimizer.zero_grad(set_to_none=True)
        use_amp = scaler is not None and scaler.is_enabled()
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            outputs = model(volumes)
            pos_logit = _pos_logit(outputs, num_classes)
            loss = criterion(pos_logit, labels)

        # --- NaN Check ---
        if torch.isnan(loss):
            print(f"\n[FATAL ERROR] NaN Loss detected at Epoch {epoch}, Batch {i}!")
            print(f"  - Logit Mean: {pos_logit.mean().item():.4f}, Std: {pos_logit.std().item():.4f}")
            print(f"  - Input Mean: {volumes.mean().item():.4f}, Std: {volumes.std().item():.4f}")
            return float('nan')

        if use_amp:
            scaler.scale(loss).backward()
            if max_grad_norm is not None and max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if max_grad_norm is not None and max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        # --- Grad Norm Check (Epoch 1 Only) ---
        if epoch == 1 and i == 0:
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            print(f"\n[DIAGNOSTIC] Epoch 1 First Batch Grad Norm: {total_norm:.4f} (Expect > 0)")

        bs = volumes.size(0)
        running_loss += float(loss.item()) * bs
        n_samples += bs

        if dbg_prob_log and not first_dbg_done:
            with torch.no_grad():
                probs = torch.sigmoid(pos_logit)
                print(f"[Log] Batch0 Probs: Mean={probs.mean().item():.3f} Std={probs.std().item():.3f} (Min={probs.min():.3f}, Max={probs.max():.3f})")
            first_dbg_done = True

        pbar.set_postfix(loss=running_loss / max(1, n_samples))

    return running_loss / max(1, n_samples)

@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int,
    criterion: nn.Module,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype = torch.float16,
) -> Tuple[float, float, float, Dict[str, Any]]:
    model.eval()
    all_probs: List[np.ndarray] = []
    all_labels: List[int] = []
    running_loss = 0.0
    n_samples = 0

    pbar = tqdm(dataloader, desc="Eval", leave=False)
    for volumes, labels in pbar:
        volumes = volumes.to(device, non_blocking=True)
        labels_t = labels.to(device, non_blocking=True).float()

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            outputs = model(volumes)
            pos_logit = _pos_logit(outputs, num_classes)
            loss = criterion(pos_logit, labels_t)

        bs = volumes.size(0)
        running_loss += float(loss.item()) * bs
        n_samples += bs
        pbar.set_postfix(val_loss=running_loss / max(1, n_samples))

        probs = torch.sigmoid(pos_logit).cpu().numpy()
        all_probs.append(probs[:, None])
        all_labels.append(labels.cpu().numpy())

    probs_concat = np.concatenate(all_probs, axis=0).reshape(-1)
    labels_concat = np.concatenate(all_labels, axis=0).astype(int)

    try:
        auc = roc_auc_score(labels_concat, probs_concat)
    except ValueError:
        warnings.warn("AUC undefined (single-class in validation); set to 0.5", UndefinedMetricWarning)
        auc = 0.5
    try:
        auprc = average_precision_score(labels_concat, probs_concat)
    except ValueError:
        warnings.warn("AUPRC undefined; set to prevalence", UndefinedMetricWarning)
        auprc = float(np.mean(labels_concat))

    ece = compute_ece(probs_concat, labels_concat)
    thr_youden = pick_threshold_by_youden(labels_concat, probs_concat)
    stats_youden = summarize_at_threshold(labels_concat, probs_concat, thr_youden)

    extras: Dict[str, Any] = {
        "auprc": float(auprc),
        "thr_youden": float(thr_youden),
    }
    for k, v in stats_youden.items():
        if k != "thr":
            extras[f"{k}@youden"] = float(v)

    val_loss = running_loss / max(1, n_samples)
    return float(auc), float(ece), float(val_loss), extras

# ----------------------- Main (single run) -----------------------
def train_teacher(
    data_root: str,
    label_csv: str,
    outdir: str,
    num_classes: int = 1,
    epochs: int = 60,
    batch_size: int = 4,
    lr: float = 1e-5, # Default set low safely
    seed: int = 42,
    val_split: float = 0.3,
    device: str | torch.device = 'cuda' if torch.cuda.is_available() else 'cpu',
    save_every: int = 5,
    freeze_bn: bool = True,
    workers: int = 4,
    weight_decay: float = 1e-4,
    max_grad_norm: float = 0.0,
    use_light_aug: bool = False,
    use_pos_weight: bool = False,
    arch: str = "resnet18",
    in_channels: int = 2,
    norm: str = "bn",
    gn_groups: int = 32,
    downsample_depth_in_layer4: bool = False,
    # --- Dataset Spatial/Crop Args ---
    crop_mode: str = "bbox",
    margin_mm: float = 6,
    target_size: Tuple[int, int, int] = (112, 144, 144),
    target_spacing: Tuple[float, float, float] = (1.5, 1.0, 1.0),
    hu_window: Tuple[float, float] = (-200.0, 250.0),
    # --- Pretrain Args ---
    pretrain_path: str | None = None,
) -> Dict[str, Any]:
    """
    Single Run Entry Point.
    """
    os.makedirs(outdir, exist_ok=True)
    set_seed(seed)
    device = torch.device(device)

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Loading Dataset...")

    # --- Dataset Initialization ---
    dataset = NiftiDataset(
        root_dir=data_root,
        label_csv=label_csv,
        transform=None,
        include_mask_channel=(in_channels >= 2),
        crop_mode=crop_mode,
        margin_mm=margin_mm,
        target_size=target_size,
        target_spacing=target_spacing,
        hu_window=hu_window,
        window_adaptive=True,
    )
    df_meta = pd.read_csv(label_csv, encoding="utf-8-sig")
    assert len(df_meta) == len(dataset), "CSV rows must match dataset length."
    labels_all = df_meta["label"].values.astype(int)
    indices = np.arange(len(dataset))

    # --- Normalization Stats ---
    sample_vol, _ = dataset[0]
    total_channels = sample_vol.shape[0]
    intensity_channels = min(1, total_channels)
    channels_to_normalize = list(range(intensity_channels))
    norm_mean = torch.zeros(total_channels, dtype=torch.float32)
    norm_std = torch.ones(total_channels, dtype=torch.float32)

    if channels_to_normalize:
        mean_est, std_est = compute_dataset_channel_stats(dataset, channels_to_normalize)
        for idx_ch, ch in enumerate(channels_to_normalize):
            norm_mean[ch] = mean_est[idx_ch]
            norm_std[ch] = std_est[idx_ch]
        print("[Normalize] channel statistics (mean/std):")
        for ch in channels_to_normalize:
            print(f"  - ch{ch}: mean={norm_mean[ch]:.6f}, std={norm_std[ch]:.6f}")
    else:
        print("[Normalize] No intensity channels found for standardisation.")

    stats_payload = {
        "mean": norm_mean,
        "std": norm_std,
        "channels": channels_to_normalize,
    }
    stats_path = os.path.join(outdir, "channel_stats.pt")
    torch.save(stats_payload, stats_path)
    print(f"[Normalize] Saved statistics to {stats_path}")
    norm_transform = ChannelNormalize(norm_mean.tolist(), norm_std.tolist(), channels=channels_to_normalize)

    # --- Stratified Split ---
    train_idx, val_idx = train_test_split(
        indices, test_size=val_split, random_state=seed, stratify=labels_all
    )

    train_labels = labels_all[train_idx]
    val_labels   = labels_all[val_idx]
    pos = int((train_labels == 1).sum())
    neg = int((train_labels == 0).sum())
    print(f"[Split] train {len(train_idx)} (pos={pos}, neg={neg}) | "
          f"val {len(val_idx)} (pos={(val_labels==1).sum()}, neg={(val_labels==0).sum()})")

    # --- Augmentation & Transforms ---
    if use_light_aug:
        aug_transform = LightAugment3D(
            p_flip=0.5, p_rotate=0.3,
            max_rotate_deg=5.0,
            p_gamma=0.0, p_contrast=0.0,
            intensity_channels=1,
        )
    else:
        aug_transform = None

    train_transform = Compose3D([aug_transform, norm_transform])
    val_transform = Compose3D([norm_transform])

    train_ds = SubsetWithTransform(dataset, train_idx, transform=train_transform)
    val_ds   = SubsetWithTransform(dataset, val_idx, transform=val_transform)

    def collate_fn(batch):
        volumes, labels_b = zip(*batch)
        volumes = torch.stack(volumes, dim=0)
        labels_b = torch.tensor(labels_b, dtype=torch.long)
        return volumes, labels_b

    # --- WeightedRandomSampler (Class Balance) ---
    class_count = np.array([(train_labels == 0).sum(), (train_labels == 1).sum()], dtype=float)
    w_per_class = 1.0 / np.maximum(class_count, 1.0)
    sample_w = np.array([w_per_class[int(l)] for l in train_labels], dtype=float)
    sampler = WeightedRandomSampler(
        torch.as_tensor(sample_w, dtype=torch.double),
        num_samples=len(sample_w),
        replacement=True
    )

    # --- DataLoaders ---
    def _seed_worker(worker_id: int):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    pin_mem = device.type == 'cuda'
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_mem,
        collate_fn=collate_fn,
        worker_init_fn=_seed_worker if workers > 0 else None,
        drop_last=True,
        persistent_workers=(workers > 0)
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_mem,
        collate_fn=collate_fn,
        worker_init_fn=_seed_worker if workers > 0 else None,
        persistent_workers=(workers > 0)
    )

    # --- Norm Layer Selection ---
    if norm.lower() == "bn":
        norm_layer = bn_factory()
    elif norm.lower() == "gn":
        norm_layer = gn_factory(gn_groups)
    elif norm.lower() == "in":
        norm_layer = in_factory(track_running_stats=False)
    else:
        raise ValueError(f"Unknown norm: {norm}. Choose from ['bn','gn','in'].")

    # --- Model Construction ---
    arch = arch.lower()
    if arch == "resnet18":
        model = generate_resnet18(num_classes=num_classes, in_channels=in_channels,
                                  downsample_depth_in_layer4=downsample_depth_in_layer4,
                                  norm_layer=norm_layer).to(device)
    elif arch == "resnet34":
        model = generate_resnet34(num_classes=num_classes, in_channels=in_channels,
                                  downsample_depth_in_layer4=downsample_depth_in_layer4,
                                  norm_layer=norm_layer).to(device)
    elif arch == "resnet50":
        model = generate_resnet50(num_classes=num_classes, in_channels=in_channels,
                                  downsample_depth_in_layer4=downsample_depth_in_layer4,
                                  norm_layer=norm_layer).to(device)
    else:
        raise ValueError(f"Unknown arch: {arch}. Choose from ['resnet18','resnet34','resnet50'].")

    # --- Pre-trained Weights Loading ---
    if pretrain_path and os.path.exists(pretrain_path):
        # 1. Load MedicalNet weights if path provided
        load_medicalnet_weights(model, pretrain_path)
    else:
        # 2. Or fallback to Inflation (Optional, or just random)
        print("[Init] No pretrain path provided or file not found. Using random init.")
        # Uncomment below if you want automatic 2D inflation when no 3D weight is given
        # try:
        #     load_pretrained_2d_weights_to_3d(model, arch=arch)
        # except Exception as e:
        #     print(f"[WARN] Failed to inflate 2D weights: {e}")

    # Freeze BN if needed (post-loading)
    if freeze_bn and norm.lower() == "bn":
        freeze_bn_running_stats(model)

    # --- Loss Function ---
    if use_pos_weight:
        pos_cnt = float((train_labels == 1).sum())
        neg_cnt = float((train_labels == 0).sum())
        pw = torch.tensor([neg_cnt / max(1.0, pos_cnt)], device=device, dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
        print(f"[INFO] Using BCEWithLogitsLoss(pos_weight={pw.item():.3f})")
    else:
        criterion = nn.BCEWithLogitsLoss()

    # --- Optimizer ---
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_bn = 'bn' in n.lower() or 'norm' in n.lower()
        if (p.ndim >= 2) and (not is_bn):
            decay.append(p)
        else:
            no_decay.append(p)
    optimizer = torch.optim.AdamW(
        [{'params': decay, 'weight_decay': weight_decay},
         {'params': no_decay, 'weight_decay': 0.0}],
        lr=lr
    )

    # --- Scheduler ---
    from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
    warmup_epochs = min(3, max(1, epochs // 20))
    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.5, total_iters=warmup_epochs),
            CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup_epochs), eta_min=lr * 0.1)
        ],
        milestones=[warmup_epochs]
    )

    best_auc = -1.0
    best_snapshot: Dict[str, Any] = {}
    history_train_loss: List[float] = []
    history_val_loss: List[float] = []

    amp_enabled = device.type == 'cuda'
    scaler = GradScaler(enabled=amp_enabled)

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Start Training...")
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, device, criterion, optimizer, num_classes,
            max_grad_norm=max_grad_norm, dbg_prob_log=(epoch == 1), freeze_bn=(freeze_bn and norm.lower()=="bn"),
            scaler=scaler, amp_dtype=torch.float16,
            epoch=epoch, outdir=outdir  # Passing for diagnostics
        )
        if math.isnan(train_loss):
            print(f"[FATAL] Train loss is NaN at epoch {epoch}. Aborting run.")
            break

        val_auc, val_ece, val_loss, extras = evaluate(
            model, val_loader, device, num_classes, criterion,
            amp_enabled=amp_enabled, amp_dtype=torch.float16,
        )

        history_train_loss.append(train_loss)
        history_val_loss.append(val_loss)

        log_msg = (
            f"[Epoch {epoch:3d}/{epochs}] "
            f"lr={_current_lr(optimizer):.2e}, "
            f"loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"val_auc={val_auc:.4f}, val_ece={val_ece:.4f}"
        )
        if "auprc" in extras:
            log_msg += f", val_auprc={extras['auprc']:.4f}"
            if "mcc@youden" in extras:
                log_msg += f", mcc@youden={extras['mcc@youden']:.3f}"
        print(log_msg)

        # [New] Periodic Feature Visualization (t-SNE)
        if epoch % save_every == 0 or epoch == epochs:
            ckpt_path = os.path.join(outdir, f"checkpoint_epoch{epoch}.pth")
            torch.save(model.state_dict(), ckpt_path)

            # Visualize Validation features
            visualize_features(model, val_loader, device, outdir, epoch, tag="val")

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), os.path.join(outdir, 'best_model.pth'))
            best_snapshot = dict(
                best_epoch=epoch,
                best_val_auc=float(val_auc),
                best_val_ece=float(val_ece),
                best_val_loss=float(val_loss),
            )
            best_snapshot.update(extras)

        scheduler.step()

    torch.save(model.state_dict(), os.path.join(outdir, 'last_model.pth'))

    # Plot Loss
    try:
        fig = plt.figure(figsize=(7.2, 4.6))
        xs = list(range(1, len(history_train_loss) + 1))
        plt.plot(xs, history_train_loss, label="train_loss")
        plt.plot(xs, history_val_loss, label="val_loss")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Train vs Val Loss")
        plt.grid(True, alpha=0.3); plt.legend()
        fig.tight_layout()
        out_png = os.path.join(outdir, "loss_curve.png")
        fig.savefig(out_png, dpi=140)
        plt.close(fig)
        print(f"[INFO] Curve saved to: {out_png}")
    except Exception as e:
        print(f"[WARN] Plot failed: {e}")

    metrics: Dict[str, Any] = {}

    def _eval_and_pack(tag: str) -> Dict[str, Any]:
        state_path = os.path.join(outdir, f"{tag}_model.pth")
        if not os.path.exists(state_path):
            return {}
        state = torch.load(state_path, map_location=device)
        model.load_state_dict(state)
        val_auc, val_ece, val_loss, extras = evaluate(
            model, val_loader, device, num_classes, criterion,
            amp_enabled=amp_enabled, amp_dtype=torch.float16,
        )
        pack = {
            f"{tag}_auc": float(val_auc),
            f"{tag}_ece": float(val_ece),
            f"{tag}_loss": float(val_loss),
        }
        pack.update({
            f"{tag}_auprc": float(extras.get("auprc", np.nan)),
            f"{tag}_thr_youden": float(extras.get("thr_youden", np.nan)),
            f"{tag}_mcc@youden": float(extras.get("mcc@youden", np.nan)),
            f"{tag}_sens@youden": float(extras.get("sens@youden", np.nan)),
            f"{tag}_spec@youden": float(extras.get("spec@youden", np.nan)),
            f"{tag}_ppv@youden": float(extras.get("ppv@youden", np.nan)),
            f"{tag}_npv@youden": float(extras.get("npv@youden", np.nan)),
            f"{tag}_f1@youden": float(extras.get("f1@youden", np.nan)),
            f"{tag}_bal_acc@youden": float(extras.get("bal_acc@youden", np.nan)),
        })
        return pack

    metrics.update(best_snapshot)
    metrics.update(_eval_and_pack("best"))
    metrics.update(_eval_and_pack("last"))

    try:
        pd.DataFrame([metrics]).to_csv(os.path.join(outdir, "metrics.csv"), index=False)
    except Exception as e:
        print(f"[WARN] Write metrics.csv failed: {e}")

    return metrics

# ----------------------- Multi-Seed Repeats -----------------------
def mean_ci(series: np.ndarray, alpha: float = 0.05) -> Tuple[float, float, float]:
    series = np.asarray(series, dtype=float)
    m = float(series.mean())
    s = float(series.std(ddof=1)) if len(series) > 1 else 0.0
    se = s / max(1, np.sqrt(len(series)))
    ci = 1.96 * se
    return m, m - ci, m + ci

# ----------------------- CLI -----------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Train 3D ResNet teacher model'
    )
    parser.add_argument('--data-root', required=True, help='Path to dataset root')
    parser.add_argument('--label-csv', required=True, help='CSV mapping case to label')
    parser.add_argument('--outdir', required=True, help='Directory to store outputs')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=4)
    # [Hardcoded Safety] Default to 1e-5 to prevent oscillation if pipeline argument fails
    parser.add_argument('--lr', type=float, default=0.00005)
    parser.add_argument('--num-classes', type=int, default=1)
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--val-split', type=float, default=0.2)
    parser.add_argument('--freeze-bn', action='store_true', default=False)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--max-grad-norm', type=float, default=0.0)
    parser.add_argument('--save-every', type=int, default=10)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--use-light-aug', action='store_true', default=False)
    parser.add_argument('--use-pos-weight', action='store_true', default=False)

    parser.add_argument('--arch', type=str, default='resnet50', choices=['resnet18','resnet34','resnet50'])
    parser.add_argument('--in-channels', type=int, default=2)
    parser.add_argument('--norm', type=str, default='bn', choices=['bn','gn','in'])
    parser.add_argument('--gn-groups', type=int, default=32)
    parser.add_argument('--downsample-depth-l4', action='store_true', default=False)

    # --- Spatial Args ---
    parser.add_argument('--crop-mode', type=str, default='bbox', choices=['bbox', 'none'])
    parser.add_argument('--margin', type=float, default=20.0)
    parser.add_argument('--target-size', type=int, nargs=3, default=[112, 144, 144])
    parser.add_argument('--target-spacing', type=float, nargs=3, default=[1.5, 1.0, 1.0])
    parser.add_argument('--hu-min', type=float, default=-200.0)
    parser.add_argument('--hu-max', type=float, default=250.0)

    # --- Pretrain Args ---
    parser.add_argument('--pretrain-path', type=str, default=None, help='Path to .pth file')

    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    # DEBUG PRINT to ensure arguments are correct
    print(f"\n{'='*40}")
    print(f"!!! DEBUG CHECK !!!")
    print(f"  - Actual LR: {args.lr:.10f}")
    print(f"  - Pretrain Path: {args.pretrain_path}")
    print(f"{'='*40}\n")

    all_metrics: List[Dict[str, Any]] = []
    for i in range(args.repeats):
        run_seed = args.seed + i
        rep_outdir = os.path.join(args.outdir, f"rep_{i+1}")
        os.makedirs(rep_outdir, exist_ok=True)

        # [NEW] Each repeat has its own full log
        rep_log_path = os.path.join(rep_outdir, "train.log")

        with tee_stdout_stderr(rep_log_path):
            print(f"\n========== Repeat {i+1}/{args.repeats} | seed={run_seed} | outdir={rep_outdir} ==========")
            print(f"[Log] This repeat log is being saved to: {rep_log_path}")

            m = train_teacher(
                data_root=args.data_root,
                label_csv=args.label_csv,
                outdir=rep_outdir,
                num_classes=args.num_classes,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                seed=run_seed,
                val_split=args.val_split,
                freeze_bn=args.freeze_bn,
                workers=args.workers,
                weight_decay=args.weight_decay,
                max_grad_norm=args.max_grad_norm,
                save_every=args.save_every,
                use_light_aug=args.use_light_aug,
                use_pos_weight=args.use_pos_weight,
                arch=args.arch,
                in_channels=args.in_channels,
                norm=args.norm,
                gn_groups=args.gn_groups,
                downsample_depth_in_layer4=args.downsample_depth_l4,
                # --- Args ---
                crop_mode=args.crop_mode,
                margin_mm=args.margin,
                target_size=tuple(args.target_size),
                target_spacing=tuple(args.target_spacing),
                hu_window=(args.hu_min, args.hu_max),
                pretrain_path=args.pretrain_path,
            )
            if 'last_auc' not in m:
                print(f"[WARN] Repeat {i+1} failed or aborted.")
                continue

            m['seed'] = run_seed
            m['repeat'] = i + 1
            all_metrics.append(m)

    if not all_metrics:
        print("[ERROR] No successful runs.")
        return

    df = pd.DataFrame(all_metrics)
    try:
        summary_rows = []

        def pick_series(col_best: str, col_last: str) -> np.ndarray:
            if col_best in df.columns:
                s = df[col_best].dropna().values
                if len(s) > 0:
                    return s
            return df[col_last].dropna().values if col_last in df.columns else np.array([])

        metrics_plan = [
            ("auc", "best_auc", "last_auc"),
            ("auprc", "best_auprc", "last_auprc"),
            ("ece", "best_ece", "last_ece"),
            ("mcc@youden", "best_mcc@youden", "last_mcc@youden"),
            ("sens@youden", "best_sens@youden", "last_sens@youden"),
            ("spec@youden", "best_spec@youden", "last_spec@youden"),
            ("ppv@youden", "best_ppv@youden", "last_ppv@youden"),
            ("npv@youden", "best_npv@youden", "last_npv@youden"),
            ("f1@youden", "best_f1@youden", "last_f1@youden"),
            ("bal_acc@youden", "best_bal_acc@youden", "last_bal_acc@youden"),
        ]

        print("\n========== Repeats Summary (mean ± 95% CI) ==========")
        for name, best_col, last_col in metrics_plan:
            arr = pick_series(best_col, last_col)
            if arr.size == 0:
                continue
            m, lo, hi = mean_ci(arr, alpha=0.05)
            summary_rows.append(dict(metric=name, mean=m, ci95_lo=lo, ci95_hi=hi, n=int(arr.size)))
            print(f"{name:15s}: {m:.4f}  [{lo:.4f}, {hi:.4f}]  (n={arr.size})")

        os.makedirs(args.outdir, exist_ok=True)
        df.to_csv(os.path.join(args.outdir, "all_runs_metrics.csv"), index=False)
        pd.DataFrame(summary_rows).to_csv(os.path.join(args.outdir, "summary.csv"), index=False)
        print(f"\n[INFO] Saved per-run metrics to all_runs_metrics.csv and summary.csv in {args.outdir}")
        print(f"[INFO] Per-repeat logs: {os.path.join(args.outdir, 'rep_k/train.log')}")
    except Exception as e:
        print(f"[WARN] Summary failed: {e}")


if __name__ == '__main__':
    main()
