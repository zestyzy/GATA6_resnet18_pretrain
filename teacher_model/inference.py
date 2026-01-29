# -*- coding: utf-8 -*-
"""
teacher_model/inference.py

Aligned with the modified train.py (incl. head-only training mode).

- Package/Script dual usage; compatible with legacy versions
- Adapts to new Dataset logic (BBox/Margin/TargetSize/HU window/window_adaptive)
- Supports probability thresholding:
    * default 0.5
    * --thr
    * --use-youden (read from metrics.csv produced by train.py)
- Supports uncertainty filtering & pseudo-labeling based on decision boundary distance:
    * --logit-q q  : use quantile of |z| as threshold; label only if |z| >= threshold
    * --logit-thr t: use absolute threshold |z| >= t
  Mutually exclusive; if enabled, final `pred_label` equals `label_logit`,
  and uncertain samples set to -1.

Output:
- <outdir>/inference.csv
- <outdir>/used_threshold.txt

Notes on alignment:
- Loads weights via robust state_dict loader (supports module. prefix and {"state_dict":...})
- Loads channel_stats.pt in the same way as train.py saves (mean/std tensors)
- Uses ChannelNormalize(channels=...) consistent with train.py behavior
- Uses include_mask_channel=(in_channels>=2) consistent with train.py
- Uses autocast inference if CUDA
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

# ---------- Compatible Import (Package/Script) ----------
try:
    from .dataset import NiftiDataset, ChannelNormalize, Compose3D
    from .resnet3d import (
        generate_resnet18, generate_resnet34, generate_resnet50,
        bn_factory, gn_factory, in_factory
    )
except Exception:
    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if pkg_root not in sys.path:
        sys.path.insert(0, pkg_root)
    from teacher_model.dataset import NiftiDataset, ChannelNormalize, Compose3D  # type: ignore
    from teacher_model.resnet3d import (  # type: ignore
        generate_resnet18, generate_resnet34, generate_resnet50,
        bn_factory, gn_factory, in_factory
    )

# ---------- Utilities ----------
def _set_seed(seed: int = 2023):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _auto_pick_files(run_dir: str) -> Dict[str, str]:
    bm = os.path.join(run_dir, "best_model.pth")
    cs = os.path.join(run_dir, "channel_stats.pt")
    if not os.path.isfile(bm):
        lm = os.path.join(run_dir, "last_model.pth")
        if os.path.isfile(lm):
            print(f"[WARN] best_model.pth not found, using last_model.pth instead.")
            bm = lm
        else:
            raise FileNotFoundError(
                f"Weights not found (checked best_model.pth and last_model.pth) in {run_dir}"
            )
    if not os.path.isfile(cs):
        raise FileNotFoundError(f"Channel stats not found: {cs}")
    return {"weights": bm, "stats": cs}

def _read_youden_threshold(run_dir: str) -> float | None:
    """
    train.py writes metrics.csv per repeat dir and also best_snapshot includes:
      - best_thr_youden (preferred)
      - thr_youden (sometimes)
      - last_thr_youden (optional)
    """
    mpath = os.path.join(run_dir, "metrics.csv")
    if not os.path.isfile(mpath):
        return None
    try:
        df = pd.read_csv(mpath, encoding="utf-8-sig")
        row = df.iloc[0].to_dict()
        for key in ["best_thr_youden", "thr_youden", "last_thr_youden"]:
            if key in row and pd.notna(row[key]):
                v = float(row[key])
                if np.isfinite(v) and 0.0 <= v <= 1.0:
                    return v
    except Exception as e:
        print(f"[WARN] Failed to read metrics.csv: {e}")
    return None

def _pos_logit(outputs: torch.Tensor, num_classes: int) -> torch.Tensor:
    # Return positive class logit
    if num_classes == 1:
        if outputs.ndim == 2 and outputs.size(1) == 1:
            return outputs.squeeze(1)
        return outputs
    return outputs[:, 1]

def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Accept caseid/case_id, imagepath/image_path, maskpath/mask_path etc."""
    lowmap = {c.lower(): c for c in df.columns}
    need_variants = {
        "case_id": ["case_id", "caseid", "case", "id"],
        "image_path": ["image_path", "imagepath", "img_path", "ct_path", "ct"],
        "mask_path": ["mask_path", "maskpath", "seg_path", "label_path", "mask"],
    }
    rename = {}
    for std, alts in need_variants.items():
        found = None
        for a in alts:
            if a in lowmap:
                found = lowmap[a]
                break
        if found is None:
            raise ValueError(f"CSV missing column: {std} (allowed aliases: {alts})")
        rename[found] = std
    return df.rename(columns=rename)

def _check_paths(df: pd.DataFrame, strict: bool = False) -> Tuple[int, int]:
    miss = 0
    for i, r in df.iterrows():
        ip, mp = str(r["image_path"]), str(r["mask_path"])
        ok_ip, ok_mp = os.path.exists(ip), os.path.exists(mp)
        if not (ok_ip and ok_mp):
            miss += 1
            print(f"[WARN] Row {i} path missing: image={ok_ip}({ip}) mask={ok_mp}({mp})")
    total = len(df)
    if miss == 0:
        print(f"[OK] Path check passed, {total} samples.")
    else:
        print(f"[WARN] {miss}/{total} samples have missing paths.")
        if strict:
            raise FileNotFoundError("Missing paths detected in strict mode.")
    return miss, total

def _load_checkpoint_state_dict(path: str, map_location="cpu") -> Dict[str, torch.Tensor]:
    """
    Robust load: supports
      - plain state_dict
      - DataParallel with 'module.' prefix
      - wrapper dict with {'state_dict': ...}
    """
    obj = torch.load(path, map_location=map_location)
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        obj = obj["state_dict"]
    if not isinstance(obj, dict):
        raise ValueError(f"Unexpected checkpoint format: {type(obj)}")

    new_state: Dict[str, torch.Tensor] = {}
    for k, v in obj.items():
        if k.startswith("module."):
            new_state[k[7:]] = v
        else:
            new_state[k] = v
    return new_state

# ---------- Head Quick Check (debug only) ----------
def _check_classifier_head(model: nn.Module, device: torch.device,
                           in_ch: int, num_classes: int,
                           input_size: Tuple[int, int, int] = (16, 64, 64)) -> None:
    """
    Quick structural check:
    - verify forward works and output shape matches expectation
    """
    name = None
    head = None
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear):
            name, head = n, m
    if head is None:
        print("[HEAD][WARN] No Linear head found, skipping check.")
        return

    D, H, W = input_size
    x = torch.randn(2, in_ch, D, H, W, device=device)
    with torch.no_grad():
        logits_full = model(x)
        z = _pos_logit(logits_full, num_classes)
    print(f"[HEAD][CHECK] Linear head='{name}', in_features={head.in_features}, out_features={head.out_features}")
    print(f"[HEAD][CHECK] Input {tuple(x.shape)} -> RawOut {tuple(logits_full.shape)} -> PosLogit {tuple(z.shape)}")

def _print_trainable_params(model: nn.Module) -> None:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[DEBUG] Params trainable={trainable} / total={total}")

# ---------- Main Logic ----------
@torch.no_grad()
def run_inference(
    csv_path: str,
    run_dir: str,
    outdir: str | None = None,
    *,
    arch: str = "resnet18",
    in_channels: int = 2,
    norm: str = "bn",
    gn_groups: int = 32,
    crop_mode: str = "bbox",
    batch_size: int = 4,
    workers: int = 4,
    num_classes: int = 1,
    # --- Spatial Params (Align with Train/Dataset) ---
    margin_mm: float = 6.0,
    target_size: Tuple[int, int, int] = (112, 144, 144),
    target_spacing: Tuple[float, float, float] = (1.5, 1.0, 1.0),
    hu_window: Tuple[float, float] = (-200.0, 250.0),
    window_adaptive: bool = True,
    # Prob threshold strategy
    thr: float | None = None,
    use_youden: bool = False,
    # logit mode
    logit_q: float | None = None,
    logit_thr: float | None = None,
    downsample_depth_in_layer4: bool = False,
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    seed: int = 2023,
    strict_paths: bool = False,
    # Debug
    strict_load: bool = True,
    head_check: bool = True,
    print_first_k: int = 0,
) -> str:
    # ---- Prep ----
    _set_seed(seed)
    device = torch.device(device)
    files = _auto_pick_files(run_dir)
    print(f"[INFO] Using weights: {files['weights']}")
    print(f"[INFO] Using channel stats: {files['stats']}")
    print(f"[INFO] Device: {device}")

    if outdir is None:
        outdir = os.path.join(run_dir, "infer_simple")
    os.makedirs(outdir, exist_ok=True)

    # ---- Threshold selection (prob) ----
    if thr is not None:
        if not (0.0 <= float(thr) <= 1.0):
            raise ValueError(f"--thr must be in [0,1], got {thr}")
        used_thr = float(thr)
        thr_src = f"--thr({thr})"
    elif use_youden:
        y = _read_youden_threshold(run_dir)
        if y is not None and 0.0 <= y <= 1.0:
            used_thr = float(y)
            thr_src = "metrics.csv:youden"
        else:
            used_thr = 0.5
            thr_src = "fallback:0.5"
    else:
        used_thr = 0.5
        thr_src = "default:0.5"
    print(f"[INFO] Prob Threshold: {used_thr:.6f} (Source: {thr_src})")

    # ---- Validate logit-mode args ----
    do_logit_mode = (logit_q is not None) ^ (logit_thr is not None)
    if (logit_q is not None) and (logit_thr is not None):
        raise ValueError("--logit-q and --logit-thr are mutually exclusive.")
    if logit_q is not None:
        q = float(logit_q)
        if not (0.0 <= q <= 1.0):
            raise ValueError("--logit-q must be in [0,1]")
    if logit_thr is not None:
        t = float(logit_thr)
        if t < 0:
            raise ValueError("--logit-thr must be non-negative")

    # ---- Read CSV ----
    print(f"[INFO] Reading inference CSV: {csv_path}")
    df_raw = pd.read_csv(csv_path, encoding="utf-8-sig")
    df = _standardize_columns(df_raw)

    # Keep user's label if exists; but inference outputs will never overwrite it.
    if "label" not in df.columns:
        df["label"] = -1

    _check_paths(df, strict=strict_paths)

    # ---- Temp CSV for NiftiDataset ----
    tmp_csv = tempfile.NamedTemporaryFile(prefix="infer_", suffix=".csv", delete=False).name
    df.to_csv(tmp_csv, index=False, encoding="utf-8-sig")

    # ---- Dataset ----
    dataset = NiftiDataset(
        root_dir="/",
        label_csv=tmp_csv,
        transform=None,
        include_mask_channel=(in_channels >= 2),
        crop_mode=crop_mode,
        margin_mm=margin_mm,
        target_size=target_size,
        target_spacing=target_spacing,
        hu_window=hu_window,
        window_adaptive=window_adaptive,  # align with train.py
    )

    # ---- Load channel stats (align with train.py) ----
    ch = torch.load(files["stats"], map_location="cpu")
    mean = ch.get("mean", torch.zeros(in_channels))
    std = ch.get("std", torch.ones(in_channels))

    if not torch.is_tensor(mean):
        mean = torch.as_tensor(mean, dtype=torch.float32)
    if not torch.is_tensor(std):
        std = torch.as_tensor(std, dtype=torch.float32)

    if mean.numel() < in_channels:
        pad = in_channels - mean.numel()
        mean = torch.cat([mean, torch.zeros(pad)], 0)
        std = torch.cat([std, torch.ones(pad)], 0)

    mean = mean[:in_channels].float()
    std = std[:in_channels].float()

    # IMPORTANT: keep channels list consistent with train.py (it only normalizes intensity channels)
    # In your train.py, you used channels_to_normalize = [0] when intensity_channels=1.
    # Here we mirror that: normalize first intensity channel only (CT), not mask.
    channels_to_normalize = [0] if in_channels >= 1 else []
    norm_t = ChannelNormalize(mean.tolist(), std.tolist(), channels=channels_to_normalize)
    test_t = Compose3D([norm_t])

    print(f"[INFO] Norm channels: {channels_to_normalize}")
    print(f"[INFO] Norm Mean (all): {mean.tolist()}")
    print(f"[INFO] Norm Std  (all): {std.tolist()}")

    class _View(Dataset):
        def __init__(self, base: Dataset, t=None):
            self.base, self.t = base, t
        def __len__(self):
            return len(self.base)
        def __getitem__(self, i):
            x, y = self.base[i]
            if self.t is not None:
                x = self.t(x)
            return x, y, i

    view = _View(dataset, test_t)

    def collate(batch):
        xs, ys, ids = zip(*batch)
        return torch.stack(xs, 0), torch.tensor(ys), torch.tensor(ids)

    loader = DataLoader(
        view,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate,
        persistent_workers=(workers > 0),
        drop_last=False,
    )
    print(f"[INFO] Samples: {len(dataset)} | Batch: {batch_size} | Workers: {workers}")

    # ---- Build Model (align with train.py factories) ----
    if norm == "bn":
        norm_layer = bn_factory()
    elif norm == "gn":
        norm_layer = gn_factory(gn_groups)
    elif norm == "in":
        norm_layer = in_factory(track_running_stats=False)
    else:
        raise ValueError("norm must be bn/gn/in")

    arch_l = arch.lower()
    if arch_l == "resnet18":
        model = generate_resnet18(
            num_classes=num_classes,
            in_channels=in_channels,
            downsample_depth_in_layer4=downsample_depth_in_layer4,
            norm_layer=norm_layer
        )
    elif arch_l == "resnet34":
        model = generate_resnet34(
            num_classes=num_classes,
            in_channels=in_channels,
            downsample_depth_in_layer4=downsample_depth_in_layer4,
            norm_layer=norm_layer
        )
    elif arch_l == "resnet50":
        model = generate_resnet50(
            num_classes=num_classes,
            in_channels=in_channels,
            downsample_depth_in_layer4=downsample_depth_in_layer4,
            norm_layer=norm_layer
        )
    else:
        raise ValueError("arch must be resnet18/resnet34/resnet50")

    # ---- Load weights robustly (align with evaluate/train changes) ----
    sd = _load_checkpoint_state_dict(files["weights"], map_location="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=strict_load)
    if (len(missing) > 0) or (len(unexpected) > 0):
        print(f"[WARN] load_state_dict strict={strict_load}: missing={len(missing)} unexpected={len(unexpected)}")
        if len(missing) > 0:
            print("  - missing (first 10):", missing[:10])
        if len(unexpected) > 0:
            print("  - unexpected (first 10):", unexpected[:10])

    model = model.to(device).eval()
    print(f"[INFO] Model: {arch_l}, Norm: {norm}, InCh: {in_channels}, Device: {device}")
    _print_trainable_params(model)

    if head_check:
        try:
            _check_classifier_head(model, device, in_channels, num_classes)
        except Exception as e:
            print(f"[HEAD][WARN] Head check failed: {e}")

    # ---- Inference Loop ----
    probs_all: List[float] = []
    logits_all: List[float] = []
    preds_prob_all: List[int] = []
    ids_all: List[int] = []
    labels_gt_all: List[int] = []

    amp = (device.type == "cuda")
    pbar = tqdm(loader, desc=f"Infer[{arch_l}]", leave=True)

    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            for vols, labels_gt, idxs in pbar:
                vols = vols.to(device, non_blocking=True)
                logits = _pos_logit(model(vols), num_classes=num_classes).float()
                probs = torch.sigmoid(logits)

                p_np = probs.detach().cpu().numpy().reshape(-1)
                z_np = logits.detach().cpu().numpy().reshape(-1)
                preds_prob = (p_np >= used_thr).astype(int)

                probs_all.extend(p_np.tolist())
                logits_all.extend(z_np.tolist())
                preds_prob_all.extend(preds_prob.tolist())
                ids_all.extend(idxs.detach().cpu().tolist())
                labels_gt_all.extend(labels_gt.detach().cpu().tolist())

                pbar.set_postfix(batch=len(vols), done=len(ids_all))

    # ---- Output Generation ----
    out_df = df.copy()

    # (1) Preserve any existing GT label as label_gt; NEVER overwrite
    if "label" in out_df.columns:
        out_df = out_df.rename(columns={"label": "label_gt"})
    else:
        out_df["label_gt"] = -1

    # (2) Create output columns
    out_df["prob_pos"] = np.nan
    out_df["pred_label"] = np.nan
    out_df["logit"] = np.nan
    out_df["abs_logit"] = np.nan

    # Safe indexing: ids_all are dataset indices (0..N-1)
    out_df.loc[ids_all, "prob_pos"] = np.array(probs_all, dtype=float)
    out_df.loc[ids_all, "pred_label"] = np.array(preds_prob_all, dtype=int)
    out_df.loc[ids_all, "logit"] = np.array(logits_all, dtype=float)
    out_df.loc[ids_all, "abs_logit"] = np.abs(out_df.loc[ids_all, "logit"].values)

    # (3) Write prob-threshold metadata (important for pipelines)
    out_df["used_threshold"] = float(used_thr)
    out_df["used_threshold_source"] = str(thr_src)

    # ===== Logit Mode =====
    used_logit_thr: Optional[float] = None
    used_logit_src = "none"

    if do_logit_mode:
        # Determine abs_logit threshold
        if logit_q is not None:
            q = float(logit_q)
            all_absz = out_df.loc[ids_all, "abs_logit"].astype(float).to_numpy()
            used_logit_thr = float(np.quantile(all_absz, q))
            used_logit_src = f"abs_logit@quantile(q={q:.3f})"
        else:
            used_logit_thr = float(logit_thr)
            used_logit_src = "abs_logit@fixed"

        keep = (out_df["abs_logit"] >= used_logit_thr).astype(int)

        # label_logit: -1 (uncertain) / 0 / 1
        label_logit = np.full(len(out_df), -1, dtype=int)

        # IMPORTANT: only assign where keep==1
        pos_mask = (out_df["logit"] >= 0) & (keep == 1)
        neg_mask = (out_df["logit"] < 0) & (keep == 1)
        label_logit[pos_mask.to_numpy()] = 1
        label_logit[neg_mask.to_numpy()] = 0

        out_df["keep_abslogit"] = keep
        out_df["label_logit"] = label_logit

        # In logit mode, final pred_label becomes label_logit (uncertain => -1)
        out_df["pred_label"] = out_df["label_logit"]

        out_df["abs_logit_thr"] = float(used_logit_thr)
        out_df["abs_logit_thr_source"] = str(used_logit_src)
    else:
        out_df["keep_abslogit"] = np.nan
        out_df["label_logit"] = np.nan
        out_df["abs_logit_thr"] = np.nan
        out_df["abs_logit_thr_source"] = "none"

    # ---- Debug: print first k rows ----
    if print_first_k and print_first_k > 0:
        k = min(int(print_first_k), len(out_df))
        print(f"\n[DEBUG] First {k} rows preview:")
        cols = ["case_id", "prob_pos", "logit", "pred_label", "label_gt"]
        cols = [c for c in cols if c in out_df.columns]
        print(out_df[cols].head(k).to_string(index=False))

    # ---- Save ----
    pos_cnt = int((out_df["pred_label"] == 1).sum())
    neg_cnt = int((out_df["pred_label"] == 0).sum())
    unk_cnt = int((out_df["pred_label"] == -1).sum()) if do_logit_mode else int(out_df["pred_label"].isna().sum())

    out_csv = os.path.join(outdir, "inference.csv")
    out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    with open(os.path.join(outdir, "used_threshold.txt"), "w", encoding="utf-8") as f:
        f.write(f"prob_thr={used_thr:.6f}\nprob_thr_source={thr_src}\n")
        if do_logit_mode and used_logit_thr is not None:
            f.write(f"abs_logit_thr={used_logit_thr:.6f}\nabs_logit_thr_source={used_logit_src}\n")
        else:
            f.write("abs_logit_thr=none\nabs_logit_thr_source=none\n")

    mode_str = "LOGIT" if do_logit_mode else "PROB"
    print(f"[DONE-{mode_str}] Results saved: {out_csv}")
    print(f"[STATS] Total={len(out_df)} | Pos={pos_cnt} | Neg={neg_cnt} | Unsure/Skipped={unk_cnt}")

    # cleanup temp
    try:
        os.remove(tmp_csv)
    except Exception:
        pass
    return out_csv

# ========== Export ==========
def generate_pseudolabels(csv_path: str, run_dir: str, outdir: str | None = None, **kwargs) -> str:
    return run_inference(csv_path=csv_path, run_dir=run_dir, outdir=outdir, **kwargs)

# ========== CLI ==========
def _build_parser():
    p = argparse.ArgumentParser(
        description="Inference (Prob threshold or Logit-based high-confidence filtering)"
    )
    p.add_argument("--csv", required=True, help="CSV with case_id, image_path, mask_path")
    p.add_argument("--run-dir", required=True, help="Train output dir (best_model.pth, channel_stats.pt)")
    p.add_argument("--out", default=None, help="Output dir (default: <run-dir>/infer_simple)")

    # Architecture
    p.add_argument("--arch", default="resnet18", choices=["resnet18", "resnet34", "resnet50"])
    p.add_argument("--in-ch", type=int, default=2, dest="in_channels")
    p.add_argument("--norm", default="bn", choices=["bn", "gn", "in"])
    p.add_argument("--gn-groups", type=int, default=32)
    p.add_argument("--crop-mode", default="bbox", choices=["bbox", "none"], help="Use bbox for 3D cropping")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--num-classes", type=int, default=1)

    # Spatial Params (align with train.py)
    p.add_argument("--margin", type=float, default=0, help="Margin in mm (default 0)")
    p.add_argument("--target-size", type=int, nargs=3, default=[112, 144, 144], help="D H W")
    p.add_argument("--target-spacing", type=float, nargs=3, default=[1.5, 1.0, 1.0], help="z y x")
    p.add_argument("--hu-min", type=float, default=-100.0)
    p.add_argument("--hu-max", type=float, default=200.0)
    p.add_argument("--no-window-adaptive", action="store_true", default=False,
                   help="Disable window_adaptive (default True to match train.py)")

    # Prob Strategy
    group_prob = p.add_mutually_exclusive_group()
    group_prob.add_argument("--thr", type=float, default=None, help="Manual prob threshold")
    group_prob.add_argument("--use-youden", action="store_true", default=False,
                            help="Use Youden threshold from metrics.csv")

    # Logit Strategy
    group_logit = p.add_mutually_exclusive_group()
    group_logit.add_argument("--logit-q", type=float, default=None,
                             help="Quantile of |z| to keep (e.g. 0.7 keeps top 30%)")
    group_logit.add_argument("--logit-thr", type=float, default=None,
                             help="Absolute |z| threshold")

    p.add_argument("--strict", action="store_true", default=False, help="Fail on missing paths")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # Debug / robustness
    p.add_argument("--strict-load", action="store_true", default=False,
                   help="Use strict=True for load_state_dict (default False to be tolerant).")
    p.add_argument("--no-head-check", action="store_true", default=False,
                   help="Disable forward sanity check with dummy input.")
    p.add_argument("--print-first-k", type=int, default=0,
                   help="Print first K rows preview after inference.")
    return p

def main(argv: List[str] | None = None):
    args = _build_parser().parse_args(argv)
    run_inference(
        csv_path=args.csv,
        run_dir=args.run_dir,
        outdir=args.out,
        arch=args.arch,
        in_channels=args.in_channels,
        norm=args.norm,
        gn_groups=args.gn_groups,
        crop_mode=args.crop_mode,
        batch_size=args.batch_size,
        workers=args.workers,
        num_classes=args.num_classes,
        # Spatial
        margin_mm=args.margin,
        target_size=tuple(args.target_size),
        target_spacing=tuple(args.target_spacing),
        hu_window=(args.hu_min, args.hu_max),
        window_adaptive=(not args.no_window_adaptive),
        # Strategy
        thr=args.thr,
        use_youden=args.use_youden,
        logit_q=args.logit_q,
        logit_thr=args.logit_thr,
        device=args.device,
        strict_paths=args.strict,
        # Debug
        strict_load=args.strict_load,
        head_check=(not args.no_head_check),
        print_first_k=args.print_first_k,
    )

if __name__ == "__main__":
    main()
