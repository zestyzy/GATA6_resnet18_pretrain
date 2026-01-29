# -*- coding: utf-8 -*-
"""
Evaluation script for trained teacher model.
Aligned with the modified train.py (incl. head-only mode).

Key alignments vs train.py:
- Uses the SAME ChannelNormalize logic (load channel_stats.pt if provided)
- Uses include_mask_channel = (in_channels >= 2)
- Uses the SAME crop/spacing/target_size/hu_window/window_adaptive arguments
- Robustly handles checkpoints saved from:
  - full finetune
  - head-only finetune (only fc updated)
  - DataParallel (module. prefix)
- [NEW] Optional debug checks:
  - verify backbone params unchanged by head-only training (by printing trainable params count)
  - dump a few per-case probabilities, and optionally save raw logits/probs
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
from typing import Tuple, Dict, Any, Optional, List

# Ensure correct imports based on package structure
try:
    from .dataset import NiftiDataset, ChannelNormalize, Compose3D
    from .resnet3d import (
        generate_resnet18, generate_resnet34, generate_resnet50,
        gn_factory, bn_factory, in_factory
    )
except ImportError:
    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if pkg_root not in sys.path:
        sys.path.insert(0, pkg_root)
    from teacher_model.dataset import NiftiDataset, ChannelNormalize, Compose3D  # type: ignore
    from teacher_model.resnet3d import (  # type: ignore
        generate_resnet18, generate_resnet34, generate_resnet50,
        gn_factory, bn_factory, in_factory
    )

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
def build_model(
    arch: str,
    num_classes: int,
    in_channels: int,
    downsample_depth_in_layer4: bool,
    norm_layer,
):
    arch = arch.lower()
    if arch == "resnet18":
        return generate_resnet18(
            num_classes=num_classes,
            in_channels=in_channels,
            downsample_depth_in_layer4=downsample_depth_in_layer4,
            norm_layer=norm_layer
        )
    elif arch == "resnet34":
        return generate_resnet34(
            num_classes=num_classes,
            in_channels=in_channels,
            downsample_depth_in_layer4=downsample_depth_in_layer4,
            norm_layer=norm_layer
        )
    elif arch == "resnet50":
        return generate_resnet50(
            num_classes=num_classes,
            in_channels=in_channels,
            downsample_depth_in_layer4=downsample_depth_in_layer4,
            norm_layer=norm_layer
        )
    raise ValueError(f"Unknown arch: {arch}")

# ----------------------- Checkpoint Loader (Aligned with train.py) -----------------------
def load_checkpoint_state_dict(model_path: str, device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Supports:
    - plain state_dict (train.py saves model.state_dict())
    - state dict with module. prefix
    - occasional wrappers (if user saved {"state_dict": ...})
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")

    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]

    if not isinstance(state, dict):
        raise ValueError(f"Unexpected checkpoint format: {type(state)}")

    new_state: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        if k.startswith("module."):
            new_state[k[7:]] = v
        else:
            new_state[k] = v
    return new_state

# ----------------------- ID extraction (robust) -----------------------
def extract_case_ids(dataset: Any) -> Optional[List[str]]:
    """
    Try to align with your train/eval dataset implementation.
    - If dataset has .samples with tuples, take first element as id/path.
    - Else return None (we will still save probs without id).
    """
    if hasattr(dataset, "samples"):
        try:
            s = getattr(dataset, "samples")
            if isinstance(s, (list, tuple)) and len(s) > 0:
                # each item expected like (case_id_or_path, label, ...)
                first = s[0]
                if isinstance(first, (list, tuple)) and len(first) > 0:
                    ids = [str(item[0]) for item in s]
                    return ids
        except Exception:
            return None
    return None

# ----------------------- Evaluation Loop -----------------------
@torch.no_grad()
def evaluate_teacher(
    model_path: str,
    data_root: str,
    label_csv: str,
    # --- Model Params (match train.py) ---
    arch: str = "resnet18",
    num_classes: int = 1,     # train.py defaults to 1 (BCE)
    in_channels: int = 2,
    norm: str = "bn",
    gn_groups: int = 32,
    downsample_depth_in_layer4: bool = False,
    # --- Data Params (must match train.py) ---
    crop_mode: str = "bbox",
    margin_mm: float = 6.0,
    target_size: Tuple[int, int, int] = (112, 144, 144),
    target_spacing: Tuple[float, float, float] = (1.5, 1.0, 1.0),
    hu_window: Tuple[float, float] = (-200.0, 250.0),
    window_adaptive: bool = True,
    # --- Runtime Params ---
    batch_size: int = 4,
    workers: int = 4,
    device: str | torch.device = 'cuda' if torch.cuda.is_available() else 'cpu',
    save_probs_csv: str | None = None,
    channel_stats_path: str | None = None,
    # --- [NEW] Debug knobs ---
    save_logits: bool = False,
    print_first_k: int = 0,
    strict_load: bool = True,
) -> Dict[str, Any]:

    device = torch.device(device)
    print(f"[Eval] Device: {device}")

    # 1) Normalization stats (aligned with train.py saving channel_stats.pt)
    norm_mean = [0.0] * in_channels
    norm_std = [1.0] * in_channels

    if channel_stats_path and os.path.exists(channel_stats_path):
        print(f"[Eval] Loading channel stats from: {channel_stats_path}")
        stats = torch.load(channel_stats_path, map_location='cpu')

        if isinstance(stats, dict):
            if "mean" in stats:
                m = stats["mean"]
                norm_mean = m.tolist() if torch.is_tensor(m) else list(m)
            if "std" in stats:
                s = stats["std"]
                norm_std = s.tolist() if torch.is_tensor(s) else list(s)

        # pad/truncate to in_channels (same spirit as your previous eval)
        if len(norm_mean) < in_channels:
            norm_mean += [0.0] * (in_channels - len(norm_mean))
        if len(norm_std) < in_channels:
            norm_std += [1.0] * (in_channels - len(norm_std))
        norm_mean = norm_mean[:in_channels]
        norm_std = norm_std[:in_channels]
    else:
        if channel_stats_path:
            print(f"[Eval] Warning: stats path not found, using mean=0 std=1: {channel_stats_path}")

    norm_transform = ChannelNormalize(norm_mean, norm_std)
    val_transform = Compose3D([norm_transform])

    # 2) Dataset (match train.py args)
    print(f"[Eval] Loading dataset from csv: {label_csv}")
    dataset = NiftiDataset(
        root_dir=data_root,
        label_csv=label_csv,
        transform=val_transform,
        include_mask_channel=(in_channels >= 2),
        crop_mode=crop_mode,
        margin_mm=margin_mm,
        target_size=target_size,
        target_spacing=target_spacing,
        hu_window=hu_window,
        window_adaptive=window_adaptive,
    )

    def collate_fn(batch):
        volumes, labels = zip(*batch)
        volumes = torch.stack(volumes, dim=0)
        labels = torch.tensor(labels, dtype=torch.long)
        return volumes, labels

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=(device.type == 'cuda'),
        collate_fn=collate_fn
    )

    case_ids = extract_case_ids(dataset)

    # 3) Build model (match train.py norm factories)
    if norm.lower() == "bn":
        norm_layer = bn_factory()
    elif norm.lower() == "gn":
        norm_layer = gn_factory(gn_groups)
    elif norm.lower() == "in":
        norm_layer = in_factory(track_running_stats=False)
    else:
        raise ValueError(f"Unknown norm: {norm}. Choose from ['bn','gn','in'].")

    model = build_model(
        arch=arch,
        num_classes=num_classes,
        in_channels=in_channels,
        downsample_depth_in_layer4=downsample_depth_in_layer4,
        norm_layer=norm_layer
    )

    # 4) Load weights
    print(f"[Eval] Loading weights from: {model_path}")
    sd = load_checkpoint_state_dict(model_path, device=device)

    # strict_load=True is safer for catching mismatch.
    # If you evaluate a head-only checkpoint on a different arch/norm, set strict_load=False.
    missing, unexpected = model.load_state_dict(sd, strict=strict_load)
    if (len(missing) > 0) or (len(unexpected) > 0):
        print(f"[Eval] load_state_dict strict={strict_load}: missing={len(missing)} unexpected={len(unexpected)}")
        if len(missing) > 0:
            print("  - missing (first 10):", missing[:10])
        if len(unexpected) > 0:
            print("  - unexpected (first 10):", unexpected[:10])

    model.to(device)
    model.eval()

    # Debug: show how many params are trainable (should be irrelevant for eval, but useful to sanity-check head-only ckpt)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Eval][Debug] Params: trainable={trainable_params} / total={total_params}")

    # 5) Inference
    all_probs: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_logits: List[np.ndarray] = []

    for i, (volumes, labels) in enumerate(loader):
        volumes = volumes.to(device, non_blocking=True)

        outputs = model(volumes)

        if num_classes == 1:
            # BCE: outputs (B,1) or (B,)
            if outputs.ndim == 2 and outputs.size(1) == 1:
                logits = outputs.squeeze(1)
            else:
                logits = outputs
            probs = torch.sigmoid(logits)
        else:
            # CE: outputs (B,2)
            logits = outputs[:, 1]
            probs = torch.softmax(outputs, dim=1)[:, 1]

        all_probs.append(probs.detach().cpu().numpy())
        all_labels.append(labels.numpy())
        if save_logits:
            all_logits.append(logits.detach().cpu().numpy())

    probs_concat = np.concatenate(all_probs, axis=0).reshape(-1)
    labels_concat = np.concatenate(all_labels, axis=0).astype(int)
    logits_concat = np.concatenate(all_logits, axis=0).reshape(-1) if save_logits else None

    # Optional: print a few cases for debug
    if print_first_k and print_first_k > 0:
        k = min(int(print_first_k), probs_concat.shape[0])
        print(f"[Eval][Debug] First {k} predictions:")
        for j in range(k):
            cid = case_ids[j] if (case_ids is not None and j < len(case_ids)) else f"idx{j}"
            if logits_concat is not None:
                print(f"  - {cid}: y={labels_concat[j]} prob={probs_concat[j]:.4f} logit={logits_concat[j]:.4f}")
            else:
                print(f"  - {cid}: y={labels_concat[j]} prob={probs_concat[j]:.4f}")

    # 6) Metrics
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

    thr_youden = pick_threshold_by_youden(labels_concat, probs_concat)
    stats = summarize_at_threshold(labels_concat, probs_concat, thr_youden)

    print("\n" + "=" * 30)
    print(f" Evaluation Results ({os.path.basename(model_path)})")
    print("=" * 30)
    print(f" AUC:       {auc:.4f}")
    print(f" AUPRC:     {auprc:.4f}")
    print(f" ECE:       {ece:.4f}")
    print(f" Threshold: {thr_youden:.4f} (Youden)")
    print(f" Bal-Acc:   {stats['bal_acc']:.4f}")
    print(f" Sens/Spec: {stats['sens']:.4f} / {stats['spec']:.4f}")
    print(f" F1:        {stats['f1']:.4f}")
    print("=" * 30 + "\n")

    metrics: Dict[str, Any] = {
        "auc": float(auc),
        "auprc": float(auprc),
        "ece": float(ece),
        **{f"{k}": float(v) for k, v in stats.items()}
    }

    # 7) Save per-case probabilities (aligned)
    if save_probs_csv:
        os.makedirs(os.path.dirname(os.path.abspath(save_probs_csv)), exist_ok=True)

        n = probs_concat.shape[0]
        if case_ids is None or len(case_ids) != n:
            if case_ids is not None:
                print(f"[Eval] Warning: case_id count {len(case_ids)} != preds {n}; will fallback to indices.")
            case_ids_out = [f"idx{i}" for i in range(n)]
        else:
            case_ids_out = case_ids

        df_out = pd.DataFrame({
            "case_id": case_ids_out,
            "label": labels_concat.astype(int),
            "prob": probs_concat,
            "pred": (probs_concat >= thr_youden).astype(int),
        })
        if save_logits and logits_concat is not None:
            df_out["logit"] = logits_concat

        df_out.to_csv(save_probs_csv, index=False)
        print(f"[Eval] Saved predictions to: {save_probs_csv}")

    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate 3D ResNet Teacher (aligned w/ train.py)')
    parser.add_argument('--data-root', required=True, help='Path to dataset root')
    parser.add_argument('--label-csv', required=True, help='CSV mapping case to label')
    parser.add_argument('--model-path', required=True, help='Path to .pth checkpoint')

    # Model params (must match training)
    parser.add_argument('--arch', default='resnet18', choices=['resnet18', 'resnet34', 'resnet50'])
    parser.add_argument('--num-classes', type=int, default=1, help='1 for BCE, 2 for CE')
    parser.add_argument('--in-channels', type=int, default=2)
    parser.add_argument('--norm', default='bn', choices=['bn', 'gn', 'in'])
    parser.add_argument('--gn-groups', type=int, default=32)
    parser.add_argument('--downsample-depth-l4', action='store_true')

    # Dataset params (must match training)
    parser.add_argument('--crop-mode', default='bbox', choices=['bbox', 'none'])
    parser.add_argument('--margin', type=float, default=0, help='Margin in mm')
    parser.add_argument('--target-size', type=int, nargs=3, default=[112, 144, 144])
    parser.add_argument('--target-spacing', type=float, nargs=3, default=[1.5, 1.0, 1.0])
    parser.add_argument('--hu-min', type=float, default=-100.0)
    parser.add_argument('--hu-max', type=float, default=200.0)
    parser.add_argument('--window-adaptive', action='store_true', default=True,
                        help='Keep consistent with train.py (default True).')

    # Eval params
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--save-probs-csv', type=str, default=None, help='Path to save detailed prediction CSV')
    parser.add_argument('--stats-path', type=str, default=None, help='Path to channel_stats.pt if available')

    # Debug
    parser.add_argument('--save-logits', action='store_true', default=False,
                        help='Also save logits to CSV (and keep logits array in memory).')
    parser.add_argument('--print-first-k', type=int, default=0,
                        help='Print first K cases with prob (and logit if enabled).')
    parser.add_argument('--strict-load', action='store_true', default=False,
                        help='Use strict=True for load_state_dict. Default False to be tolerant.')

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
        crop_mode=args.crop_mode,
        margin_mm=args.margin,
        target_size=tuple(args.target_size),
        target_spacing=tuple(args.target_spacing),
        hu_window=(args.hu_min, args.hu_max),
        window_adaptive=args.window_adaptive,
        batch_size=args.batch_size,
        workers=args.workers,
        save_probs_csv=args.save_probs_csv,
        channel_stats_path=args.stats_path,
        save_logits=args.save_logits,
        print_first_k=args.print_first_k,
        strict_load=args.strict_load,
    )
