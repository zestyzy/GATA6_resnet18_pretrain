# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import os
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Callable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

# -------------------------
# 1. NIfTI I/O & Resampling (包含关键修复)
# -------------------------
def _load_nifti_with_spacing(path: str) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """
    Load a NIfTI file and return (data, spacing_mm).
    
    [CRITICAL FIXES]:
    1. Force convert to canonical orientation (RAS+) to ensure Image and Mask alignment.
    2. Transpose data from (W, H, D) to (D, H, W) for PyTorch processing.
    """
    try:
        import nibabel as nib
    except ImportError as e:
        raise ImportError(
            "nibabel is required. Please install with: pip install nibabel"
        ) from e
    
    img = nib.load(path)
    
    # --- [Fix 1] 强制转换为标准物理方向 (RAS+)，解决 Mask 翻转/错位问题 ---
    img = nib.as_closest_canonical(img)
    
    data = img.get_fdata().astype(np.float32)
    
    # nibabel header zooms are usually (dx, dy, dz) matching the (W, H, D) raw data
    zooms = img.header.get_zooms()[:3]
    
    # --- [Fix 2] 维度转置: (W, H, D) -> (D, H, W) ---
    # PyTorch 3D layers expect (Depth, Height, Width) -> (z, y, x)
    if data.ndim == 3:
        data = data.transpose(2, 1, 0)
        # spacing 也随之调整顺序: (dx, dy, dz) -> (dz, dy, dx)
        spacing = (float(zooms[2]), float(zooms[1]), float(zooms[0]))
    else:
        # Fallback (rarely used for 3D volumes)
        spacing = (float(zooms[2]), float(zooms[1]), float(zooms[0]))

    return data, spacing

def _round_int(x: float) -> int:
    return int(round(float(x)))

def _spacing_to_new_size(
    in_shape: Tuple[int, int, int],
    in_spacing: Tuple[float, float, float],
    out_spacing: Tuple[float, float, float],
) -> Tuple[int, int, int]:
    """
    Calculate new shape (D,H,W) based on physical spacing ratio.
    """
    Dz, Dy, Dx = in_shape
    sz_in, sy_in, sx_in = in_spacing
    sz_out, sy_out, sx_out = out_spacing
    new_D = _round_int(Dz * (sz_in / sz_out))
    new_H = _round_int(Dy * (sy_in / sy_out))
    new_W = _round_int(Dx * (sx_in / sx_out))
    return max(1, new_D), max(1, new_H), max(1, new_W)

def _resample3d_torch(
    vol_np: np.ndarray,
    new_size: Tuple[int, int, int],
    mode: str = "trilinear",
) -> np.ndarray:
    """
    Resample (D,H,W) array to new_size using PyTorch interpolation.
    """
    # (D,H,W) -> (1,1,D,H,W)
    vol = torch.from_numpy(vol_np)[None, None]
    if mode == "nearest":
        out = F.interpolate(vol, size=new_size, mode="nearest")
    else:
        out = F.interpolate(vol, size=new_size, mode="trilinear", align_corners=False)
    # (1,1,D,H,W) -> (D,H,W)
    return out.squeeze(0).squeeze(0).cpu().numpy()

# -------------------------
# 2. Geometry / BBox Helpers
# -------------------------
def compute_bounding_box(mask: np.ndarray) -> Tuple[slice, slice, slice]:
    """
    Compute minimal bounding box of non-zero elements in mask.
    Returns slices (slice(z_min, z_max), ...).
    """
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        D, H, W = mask.shape
        return slice(0, D), slice(0, H), slice(0, W)
    
    zmin, ymin, xmin = coords.min(axis=0)
    zmax, ymax, xmax = coords.max(axis=0)
    # +1 because slice stop is exclusive
    return slice(zmin, zmax + 1), slice(ymin, ymax + 1), slice(xmin, xmax + 1)

def expand_bounding_box_vox(
    slices: Tuple[slice, slice, slice],
    margin_vox: Tuple[int, int, int],
    volume_shape: Tuple[int, int, int],
) -> Tuple[slice, slice, slice]:
    """
    Expand bounding box by margin_vox (z, y, x) with boundary clamping.
    """
    (z_sl, y_sl, x_sl) = slices
    mz, my, mx = margin_vox
    D, H, W = volume_shape
    
    z0 = max(0, (z_sl.start or 0) - mz)
    y0 = max(0, (y_sl.start or 0) - my)
    x0 = max(0, (x_sl.start or 0) - mx)
    
    z1 = min(D, (z_sl.stop or D) + mz)
    y1 = min(H, (y_sl.stop or H) + my)
    x1 = min(W, (x_sl.stop or W) + mx)
    
    return slice(z0, z1), slice(y0, y1), slice(x0, x1)

# -------------------------
# 3. Intensity Helpers
# -------------------------
def _normalize_hu_to_m11(x: np.ndarray, hu_min: float, hu_max: float) -> np.ndarray:
    """
    Clip values to [hu_min, hu_max] and linearly map to [-1, 1].
    """
    x = np.clip(x, hu_min, hu_max)
    denom = max(hu_max - hu_min, 1e-6)
    x = 2.0 * (x - hu_min) / denom - 1.0
    return x.astype(np.float32)

# -------------------------
# 4. Augmentation & Transform
# -------------------------
class LightAugment3D:
    """
    Lightweight geometric augmentation.
    [UPDATED] Now supports Gaussian Noise injection for NCCT robustness.
    """
    def __init__(
        self,
        p_flip: float = 0.5,
        p_rotate: float = 0.5,
        p_gamma: float = 0.5,
        p_contrast: float = 0.5,
        p_noise: float = 0.5,          # [新增] 噪声概率
        max_rotate_deg: float = 5.0,
        max_gamma_delta: float = 0.10,
        max_contrast_delta: float = 0.10,
        max_noise_std: float = 0.05,   # [新增] 噪声标准差 (相对于 [-1, 1] 范围)
        padding_mode: str = "border",
        intensity_channels: int = 1,
    ) -> None:
        self.p_flip = float(p_flip)
        self.p_rotate = float(p_rotate)
        self.p_gamma = float(p_gamma)
        self.p_contrast = float(p_contrast)
        self.p_noise = float(p_noise)
        
        self.max_rotate_deg = float(max_rotate_deg)
        self.max_gamma_delta = float(max_gamma_delta)
        self.max_contrast_delta = float(max_contrast_delta)
        self.max_noise_std = float(max_noise_std)
        
        self.padding_mode = padding_mode
        self.intensity_channels = int(max(0, intensity_channels))

    def _maybe(self, p: float) -> bool:
        return random.random() < p

    def _rotate_inplane(self, vol: torch.Tensor, deg: float) -> torch.Tensor:
        if abs(deg) < 1e-6:
            return vol
        device = vol.device
        orig_dtype = vol.dtype
        rad = math.radians(deg)
        c, s = math.cos(rad), math.sin(rad)
        # Affine matrix for 2D rotation around Z axis in 3D
        theta = torch.tensor([[[ c, -s, 0.0, 0.0],
                               [ s,  c, 0.0, 0.0],
                               [0.0, 0.0, 1.0, 0.0]]], device=device, dtype=torch.float32)
        vol5 = vol.unsqueeze(0).to(dtype=torch.float32)
        grid = F.affine_grid(theta, size=vol5.size(), align_corners=False)
        vol_rot = F.grid_sample(vol5, grid, mode='bilinear', padding_mode=self.padding_mode, align_corners=False)
        return vol_rot.squeeze(0).to(dtype=orig_dtype)

    def _gamma_jitter(self, vol: torch.Tensor, max_delta: float) -> torch.Tensor:
        # Map [-1, 1] to [0, 1] for gamma
        x01 = ((vol + 1.0) * 0.5).clamp(0.0, 1.0)
        gamma = 1.0 + random.uniform(-max_delta, max_delta)
        x01 = x01.pow(gamma)
        out = x01 * 2.0 - 1.0
        return out.clamp(-1.0, 1.0)

    def _contrast_jitter(self, vol: torch.Tensor, max_delta: float) -> torch.Tensor:
        factor = 1.0 + random.uniform(-max_delta, max_delta)
        out = vol * factor
        return out.clamp(-1.0, 1.0)

    def _gaussian_noise(self, vol: torch.Tensor, max_std: float) -> torch.Tensor:
        """
        [新增] 添加高斯噪声
        """
        std = random.uniform(0, max_std)
        noise = torch.randn_like(vol) * std
        out = vol + noise
        return out.clamp(-1.0, 1.0)

    def __call__(self, vol: torch.Tensor) -> torch.Tensor:
        if vol.ndim != 4:
            raise ValueError(f"Expected (C,D,H,W) tensor, got {tuple(vol.shape)}")
        
        # Geometric transforms (apply to ALL channels including mask)
        if self._maybe(self.p_flip):
            vol = vol.flip(-1) # Flip width
        if self._maybe(self.p_rotate):
            deg = random.uniform(-self.max_rotate_deg, self.max_rotate_deg)
            vol = self._rotate_inplane(vol, deg)

        # Intensity transforms (apply ONLY to intensity channels)
        if self.intensity_channels > 0:
            ch = min(self.intensity_channels, vol.size(0))
            intensity = vol[:ch]
            
            # [关键] 噪声注入：防止模型过拟合 NCCT 的微观噪声纹理
            if self._maybe(self.p_noise):
                intensity = self._gaussian_noise(intensity, self.max_noise_std)

            if self._maybe(self.p_gamma):
                intensity = self._gamma_jitter(intensity, self.max_gamma_delta)
            if self._maybe(self.p_contrast):
                intensity = self._contrast_jitter(intensity, self.max_contrast_delta)
            
            if ch == vol.size(0):
                vol = intensity
            else:
                # 重新拼回 Mask
                vol = torch.cat([intensity, vol[ch:]], dim=0)
        return vol

class Compose3D:
    def __init__(self, transforms: Sequence[Optional[Callable]]):
        self.transforms = [t for t in transforms if t is not None]

    def __call__(self, vol: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            vol = t(vol)
        return vol

class ChannelNormalize:
    def __init__(self, mean, std, channels=None, eps=1e-6):
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32).clamp(min=float(eps))
        self.channels = list(range(len(mean))) if channels is None else list(channels)

    def __call__(self, vol: torch.Tensor) -> torch.Tensor:
        device = vol.device
        mean = self.mean.to(device)
        std = self.std.to(device)
        for ch in self.channels:
            if ch < vol.size(0):
                vol[ch] = (vol[ch] - mean[ch]) / std[ch]
        return vol

# -------------------------
# 5. Helpers for File Discovery
# -------------------------
def _find_case_dir_by_prefix(root_dir: Path, case_id: str) -> Optional[Path]:
    if not case_id: return None
    for p in sorted(root_dir.iterdir()):
        if p.is_dir() and p.name.startswith(str(case_id)):
            return p
    return None

def _find_nifti_pair(case_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    img, msk = None, None
    for p in sorted(case_dir.iterdir()):
        n = p.name.lower()
        if not (n.endswith(".nii") or n.endswith(".nii.gz")):
            continue
        if ("image" in n) or ("img" in n):
            img = p
        elif ("label" in n) or ("mask" in n):
            msk = p
    return img, msk

# -------------------------
# 6. Main Dataset Class
# -------------------------
class NiftiDataset(Dataset):
    """
    Complete NIfTI Dataset implementing the logic:
    1. Physical Resampling -> 2. BBox Crop (Margin) -> 3. Resize to Target Size -> 4. 2-Channel Output
    """
    def __init__(
        self,
        root_dir: str,
        label_csv: str,
        target_size: Tuple[int, int, int] = (112, 224, 224),
        target_spacing: Tuple[float, float, float] = (1.5, 1.0, 1.0),
        margin_mm: float = 20.0,
        crop_mode: str = "bbox",  # 'bbox' = A3 logic (Lesion+Margin), 'none' = A4 logic (Whole Volume)
        hu_window: Tuple[float, float] = (-150.0, 250.0),
        transform: Optional[Callable] = None,
        include_mask_channel: bool = True,
        window_adaptive: bool = True,
        window_percentiles: Tuple[float, float] = (0.5, 99.5),
    ) -> None:
        self.root_dir = Path(root_dir)
        self.target_size = tuple(target_size)
        self.target_spacing = tuple(float(x) for x in target_spacing)
        self.margin_mm = float(margin_mm)
        self.crop_mode = str(crop_mode).lower()
        self.hu_min, self.hu_max = hu_window
        self.transform = transform
        self.include_mask_channel = include_mask_channel
        self.window_adaptive = window_adaptive
        
        # Safe percentile sorting
        p0, p1 = window_percentiles
        self.window_percentiles = (min(p0, p1), max(p0, p1))

        # ---- Load CSV ----
        self.df = pd.read_csv(label_csv, encoding="utf-8-sig")
        # Clean columns (remove BOM, strip spaces)
        clean_cols = {c: str(c).strip().lstrip("\ufeff") for c in self.df.columns}
        self.df = self.df.rename(columns=clean_cols)
        
        # Identify columns
        lower_cols = {c.lower(): c for c in self.df.columns}
        
        # Case ID
        id_col = None
        for k in ["case_id", "case"]:
            if k in self.df.columns: id_col = k; break
            if k in lower_cols: id_col = lower_cols[k]; break
        if not id_col: raise KeyError("CSV missing 'case_id' column")

        # Label
        lbl_col = None
        for k in ["label"]:
            if k in self.df.columns: lbl_col = k; break
            if k in lower_cols: lbl_col = lower_cols[k]; break
        if not lbl_col: raise KeyError("CSV missing 'label' column")

        img_col = lower_cols.get("image_path")
        msk_col = lower_cols.get("mask_path")

        # Build sample list
        self.samples = []
        for _, row in self.df.iterrows():
            cid = str(row[id_col]).strip()
            lbl = int(row[lbl_col])
            
            # Path discovery: CSV absolute paths > Folder search
            i_p, m_p = None, None
            if img_col and msk_col:
                i_p = str(row[img_col]).strip()
                m_p = str(row[msk_col]).strip()
            else:
                c_dir = _find_case_dir_by_prefix(self.root_dir, cid)
                if c_dir:
                    p1, p2 = _find_nifti_pair(c_dir)
                    if p1 and p2:
                        i_p, m_p = str(p1), str(p2)
            
            if i_p and m_p and os.path.exists(i_p) and os.path.exists(m_p):
                self.samples.append((cid, i_p, m_p, lbl))
            else:
                # You might want to warn here, but for now we skip invalid paths
                pass

        if not self.samples:
            raise RuntimeError(f"No valid samples found in {label_csv}")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_case(self, img_path: str, mask_path: str) -> torch.Tensor:
        img_roi, msk_roi = self._prepare_roi(img_path, mask_path)
        wmin, wmax = self._compute_window_params(img_roi, msk_roi)
        img_roi = _normalize_hu_to_m11(img_roi, wmin, wmax)
        msk_roi = (msk_roi > 0).astype(np.float32)

        # 5. Final Resize to Fixed Tensor Shape (Interpolate)
        #    This effectively "zooms" the cropped ROI to fill the target volume.
        img_t = torch.from_numpy(img_roi).unsqueeze(0).unsqueeze(0) # (1,1,D,H,W)
        img_t = F.interpolate(img_t, size=self.target_size, mode="trilinear", align_corners=False).squeeze(0) # (1,D,H,W)

        if self.include_mask_channel:
            msk_t = torch.from_numpy(msk_roi).unsqueeze(0).unsqueeze(0)
            msk_t = F.interpolate(msk_t, size=self.target_size, mode="nearest").squeeze(0)
            # Concatenate: Channel 0 = Image, Channel 1 = Mask
            vol = torch.cat([img_t, msk_t], dim=0) # (2,D,H,W)
        else:
            vol = img_t

        return vol

    def _prepare_roi(self, img_path: str, mask_path: str) -> Tuple[np.ndarray, np.ndarray]:
        # 1. Load Data & Spacing (Align orientation)
        img_np, spacing_in = _load_nifti_with_spacing(img_path)
        msk_np, _ = _load_nifti_with_spacing(mask_path)

        # 2. Resample to Target Physical Spacing (e.g. 1.5mm)
        #    Crucial so that 'margin_mm' means the same physical distance for everyone.
        new_size = _spacing_to_new_size(img_np.shape, spacing_in, self.target_spacing)
        img_rs = _resample3d_torch(img_np, new_size, mode="trilinear")
        msk_rs = _resample3d_torch(msk_np, new_size, mode="nearest")

        # 3. Crop Logic (ROI vs Whole)
        if self.crop_mode == "bbox":
            # Compute BBox from Mask
            bbox = compute_bounding_box(msk_rs)
            # Convert physical margin (mm) to voxel margin
            mz = _round_int(self.margin_mm / self.target_spacing[0])
            my = _round_int(self.margin_mm / self.target_spacing[1])
            mx = _round_int(self.margin_mm / self.target_spacing[2])
            # Expand BBox
            bbox = expand_bounding_box_vox(bbox, (mz, my, mx), img_rs.shape)

            z, y, x = bbox
            img_roi = img_rs[z, y, x]
            msk_roi = msk_rs[z, y, x]
        else:
            # None mode: Use whole volume (A4)
            img_roi = img_rs
            msk_roi = msk_rs

        return img_roi, msk_roi

    def _compute_window_params(self, img_roi: np.ndarray, msk_roi: np.ndarray) -> Tuple[float, float]:
        # Adaptive windowing based on the ROI computed above
        if self.window_adaptive:
            valid_mask = (msk_roi > 0.5)
            if valid_mask.any():
                roi_pixels = img_roi[valid_mask]
                try:
                    wmin = float(np.percentile(roi_pixels, self.window_percentiles[0]))
                    wmax = float(np.percentile(roi_pixels, self.window_percentiles[1]))
                    # Safety check for degenerate window
                    if wmax - wmin < 1e-3:
                        wmin, wmax = self.hu_min, self.hu_max
                except Exception:
                    wmin, wmax = self.hu_min, self.hu_max
            else:
                # If no lesion mask is available, fall back to fixed HU window
                wmin, wmax = self.hu_min, self.hu_max
        else:
            wmin, wmax = self.hu_min, self.hu_max

        return wmin, wmax

    def compute_window_stats(self, idx: int, hist_bins: int | None = None) -> Dict[str, np.ndarray | float]:
        cid, i_p, m_p, lbl = self.samples[idx]
        img_roi, msk_roi = self._prepare_roi(i_p, m_p)
        wmin, wmax = self._compute_window_params(img_roi, msk_roi)
        img_win = _normalize_hu_to_m11(img_roi, wmin, wmax)
        stats: Dict[str, np.ndarray | float] = {
            "case_id": cid,
            "label": int(lbl),
            "wmin": float(wmin),
            "wmax": float(wmax),
            "mean": float(np.mean(img_win)),
            "std": float(np.std(img_win)),
        }
        if hist_bins is not None and hist_bins > 0:
            hist, bin_edges = np.histogram(img_win, bins=int(hist_bins), range=(-1.0, 1.0))
            stats["hist"] = hist.astype(np.int64)
            stats["bin_edges"] = bin_edges.astype(np.float32)
        return stats

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        cid, i_p, m_p, lbl = self.samples[idx]
        vol = self._load_case(i_p, m_p)
        if self.transform:
            vol = self.transform(vol)
        return vol, lbl
