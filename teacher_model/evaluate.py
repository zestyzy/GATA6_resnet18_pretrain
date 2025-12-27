# -*- coding: utf-8 -*-
"""
Evaluation script for trained teacher model.
Compatible with the enhanced dataset.py (ROI cropping, Resizing, etc.)

Example:
python -m teacher_model.evaluate \
  --data-root /path/to/test_data \
  --label-csv /path/to/test.csv \
  --model-path /path/to/best_model.pth \
  --arch resnet18 \
  --crop-mode bbox --margin 6.0 \
  --save-probs-csv /path/to/results.csv
"""

from __future__ import annotations

import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve,
    f1_score, matthews_corrcoef, balanced_accuracy_score
)
from sklearn.exceptions import UndefinedMetricWarning
import warnings
import pandas as pd
from typing import Tuple, Dict, Any

# Ensure correct imports based on package structure
try:
    from .dataset import NiftiDataset, ChannelNormalize, Compose3D
    from .resnet3d import generate_resnet18, generate_resnet34, generate_resnet50, gn_factory, bn_factory, in_factory
except ImportError:
    # Fallback for running as script
    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if pkg_root not in sys.path:
        sys.path.insert(0, pkg_root)
    from teacher_model.dataset import NiftiDataset, ChannelNormalize, Compose3D # type: ignore
    from teacher_model.resnet3d import generate_resnet18, generate_resnet34, generate_resnet50, gn_factory, bn_factory, in_factory # type: ignore

# ----------------------- Metrics Helpers (Reuse train.py logic) -----------------------
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

# ----------------------- Model Builder -----------------------
def build_model(arch: str, num_classes: int, in_channels: int,
                downsample_depth_in_layer4: bool, norm_layer):
    arch = arch.lower()
    if arch == "resnet18":
        return generate_resnet18(num_classes=num_classes, in_channels=in_channels,
                                 downsample_depth_in_layer4=downsample_depth_in_layer4,
                                 norm_layer=norm_layer)
    elif arch == "resnet34":
        return generate_resnet34(num_classes=num_classes, in_channels=in_channels,
                                 downsample_depth_in_layer4=downsample_depth_in_layer4,
                                 norm_layer=norm_layer)
    elif arch == "resnet50":
        return generate_resnet50(num_classes=num_classes, in_channels=in_channels,
                                 downsample_depth_in_layer4=downsample_depth_in_layer4,
                                 norm_layer=norm_layer)
    raise ValueError(f"Unknown arch: {arch}")

# ----------------------- Evaluation Loop -----------------------
@torch.no_grad()
def evaluate_teacher(
    model_path: str,
    data_root: str,
    label_csv: str,
    # --- Model Params ---
    arch: str = "resnet18",
    num_classes: int = 1,     # train.py defaults to 1 (BCE)
    in_channels: int = 2,
    norm: str = "gn",
    gn_groups: int = 32,
    downsample_depth_in_layer4: bool = False,
    # --- Data Params (Must match train.py) ---
    crop_mode: str = "bbox",
    margin_mm: float = 6.0,
    target_size: Tuple[int, int, int] = (112, 144, 144),
    target_spacing: Tuple[float, float, float] = (1.5, 1.0, 1.0),
    hu_window: Tuple[float, float] = (-200.0, 250.0), # Added HU window
    # --- Runtime Params ---
    batch_size: int = 4,
    workers: int = 4,
    device: str | torch.device = 'cuda' if torch.cuda.is_available() else 'cpu',
    save_probs_csv: str | None = None,
    channel_stats_path: str | None = None, # Path to saved mean/std
) -> Dict[str, Any]:
    
    device = torch.device(device)
    print(f"[Eval] Device: {device}")
    
    # 1. Normalization Stats
    # If stats path provided, load it. Else default to 0/1 (Dataset already does HU->[-1,1])
    norm_mean = [0.0] * in_channels
    norm_std = [1.0] * in_channels
    
    if channel_stats_path and os.path.exists(channel_stats_path):
        print(f"[Eval] Loading channel stats from {channel_stats_path}")
        stats = torch.load(channel_stats_path, map_location='cpu')
        # Simple check for 'mean' and 'std' keys
        if "mean" in stats:
            m = stats["mean"]
            norm_mean = m.tolist() if torch.is_tensor(m) else m
            # Pad if channels mismatch (e.g., stats saved for 1ch but inferring 2ch)
            if len(norm_mean) < in_channels:
                norm_mean += [0.0] * (in_channels - len(norm_mean))
                
        if "std" in stats:
            s = stats["std"]
            norm_std = s.tolist() if torch.is_tensor(s) else s
            if len(norm_std) < in_channels:
                norm_std += [1.0] * (in_channels - len(norm_std))

    norm_transform = ChannelNormalize(norm_mean[:in_channels], norm_std[:in_channels])
    val_transform = Compose3D([norm_transform])

    # 2. Build Dataset
    # CRITICAL: Pass crop_mode, margin_mm, etc. to ensure FOV matches training
    print(f"[Eval] Loading dataset from {label_csv}")
    dataset = NiftiDataset(
        root_dir=data_root,
        label_csv=label_csv,
        transform=val_transform, 
        target_size=target_size,
        target_spacing=target_spacing,
        margin_mm=margin_mm,
        crop_mode=crop_mode,
        include_mask_channel=(in_channels >= 2),
        hu_window=hu_window, # Pass HU window
        window_adaptive=True, # Keep consistent with training
    )

    def collate_fn(batch):
        volumes, labels = zip(*batch)
        volumes = torch.stack(volumes, dim=0)
        labels = torch.tensor(labels, dtype=torch.long)
        return volumes, labels

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=(device.type == 'cuda'), 
        collate_fn=collate_fn
    )

    # 3. Build Model
    if norm.lower() == "bn":
        norm_layer = bn_factory()
    elif norm.lower() == "gn":
        norm_layer = gn_factory(gn_groups)
    elif norm.lower() == "in":
        norm_layer = in_factory(track_running_stats=False)
    else:
        norm_layer = bn_factory()

    model = build_model(arch, num_classes, in_channels, downsample_depth_in_layer4, norm_layer)
    
    # Load Weights
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")
    
    print(f"[Eval] Loading weights from {model_path}")
    state = torch.load(model_path, map_location=device)
    # Handle potential DataParallel 'module.' prefix
    new_state = {}
    for k, v in state.items():
        if k.startswith("module."):
            new_state[k[7:]] = v
        else:
            new_state[k] = v
    model.load_state_dict(new_state)
    model.to(device)
    model.eval()

    # 4. Inference Loop
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    
    # Dataset order is preserved in sequential loader
    case_ids = [s[0] for s in dataset.samples] 

    for i, (volumes, labels) in enumerate(loader):
        volumes = volumes.to(device, non_blocking=True)
        
        with torch.no_grad():
            outputs = model(volumes)
            
            # Compatible with BCE (num_classes=1) and CE (num_classes=2)
            if num_classes == 1:
                # BCE: Output shape (B, 1) or (B,) -> Sigmoid
                if outputs.ndim == 2 and outputs.size(1) == 1:
                    logits = outputs.squeeze(1)
                else:
                    logits = outputs
                probs = torch.sigmoid(logits)
            else:
                # CE: Output shape (B, 2) -> Softmax -> Take column 1
                probs = torch.softmax(outputs, dim=1)[:, 1]

        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.numpy())

    probs_concat = np.concatenate(all_probs, axis=0)
    labels_concat = np.concatenate(all_labels, axis=0)

    # 5. Compute Metrics
    try:
        auc = roc_auc_score(labels_concat, probs_concat)
    except ValueError:
        warnings.warn("AUC undefined (single-class in eval set)", UndefinedMetricWarning)
        auc = 0.5
    
    try:
        auprc = average_precision_score(labels_concat, probs_concat)
    except ValueError:
        auprc = float(np.mean(labels_concat))

    ece = compute_ece(probs_concat, labels_concat)
    
    # Calculate Youden threshold & detailed stats
    thr_youden = pick_threshold_by_youden(labels_concat, probs_concat)
    stats = summarize_at_threshold(labels_concat, probs_concat, thr_youden)

    print("\n" + "="*30)
    print(f" Evaluation Results ({os.path.basename(model_path)})")
    print("="*30)
    print(f" AUC:      {auc:.4f}")
    print(f" AUPRC:    {auprc:.4f}")
    print(f" ECE:      {ece:.4f}")
    print(f" Threshold:{thr_youden:.4f} (Youden)")
    print(f" Accuracy: {stats['bal_acc']:.4f} (Balanced)")
    print(f" Sens/Spec:{stats['sens']:.4f} / {stats['spec']:.4f}")
    print(f" F1-Score: {stats['f1']:.4f}")
    print("="*30 + "\n")

    metrics = {
        "auc": float(auc),
        "auprc": float(auprc),
        "ece": float(ece),
        **{f"{k}": float(v) for k, v in stats.items()}
    }

    # 6. Save Results
    if save_probs_csv:
        # Ensure lengths match
        if len(case_ids) != len(probs_concat):
            print(f"[Warn] ID count ({len(case_ids)}) != Preds ({len(probs_concat)}). IDs may be mismatched.")
        
        limit = min(len(case_ids), len(probs_concat))
        df_out = pd.DataFrame({
            "case_id": case_ids[:limit],
            "label": labels_concat[:limit].astype(int),
            "prob": probs_concat[:limit],
            "pred": (probs_concat[:limit] >= thr_youden).astype(int)
        })
        os.makedirs(os.path.dirname(os.path.abspath(save_probs_csv)), exist_ok=True)
        df_out.to_csv(save_probs_csv, index=False)
        print(f"[INFO] Saved predictions to {save_probs_csv}")

    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate 3D ResNet Teacher')
    parser.add_argument('--data-root', required=True, help='Path to dataset root')
    parser.add_argument('--label-csv', required=True, help='CSV mapping case to label')
    parser.add_argument('--model-path', required=True, help='Path to .pth checkpoint')
    
    # Model params (Must match training)
    parser.add_argument('--arch', default='resnet18', choices=['resnet18','resnet34','resnet50'])
    parser.add_argument('--num-classes', type=int, default=1, help='1 for BCE, 2 for CE')
    parser.add_argument('--in-channels', type=int, default=2)
    parser.add_argument('--norm', default='gn', choices=['bn','gn','in'])
    parser.add_argument('--gn-groups', type=int, default=32)
    parser.add_argument('--downsample-depth-l4', action='store_true')
    
    # Dataset params (Must match training)
    parser.add_argument('--crop-mode', default='bbox', choices=['bbox', 'none'])
    parser.add_argument('--margin', type=float, default=6.0, help='Margin in mm (default 6.0)')
    parser.add_argument('--target-size', type=int, nargs=3, default=[112, 144, 144])
    parser.add_argument('--target-spacing', type=float, nargs=3, default=[1.5, 1.0, 1.0])
    parser.add_argument('--hu-min', type=float, default=-200.0)
    parser.add_argument('--hu-max', type=float, default=250.0)
    
    # Eval params
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--save-probs-csv', type=str, default=None, help='Path to save detailed prediction CSV')
    parser.add_argument('--stats-path', type=str, default=None, help='Path to channel_stats.pt if available')
    
    args = parser.parse_args()

    evaluate_teacher(
        model_path=args.model_path,
        data_root=args.data_root,
        label_csv=args.label_csv,
        arch=args.arch,
        num_classes=args.num_classes,
        in_channels=args.in_channels,
        norm=args.norm,
        gn_groups=args.gn_groups,
        downsample_depth_in_layer4=args.downsample_depth_l4,
        # Spatial params
        crop_mode=args.crop_mode,
        margin_mm=args.margin,
        target_size=tuple(args.target_size),
        target_spacing=tuple(args.target_spacing),
        hu_window=(args.hu_min, args.hu_max),
        # Runtime params
        batch_size=args.batch_size,
        workers=args.workers,
        save_probs_csv=args.save_probs_csv,
        channel_stats_path=args.stats_path
    )