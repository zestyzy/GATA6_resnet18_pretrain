#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exact_from_csv.py —— 批量提取 CT 组学特征（标准化：重采样1mm + [可选]bbox 裁剪 + ROI内 z-score）
- 不做 HU 窗口
- 不输出 diagnostics_* 列
- 默认不输出 pre_* 元信息（用 --keep-meta 可保留）
- 量化方式可选：--bin-count（默认=64，推荐配合 z-score）或 --bin-width（二选一，互斥）
- 显式关闭 PyRadiomics 内部 normalize / resampledPixelSpacing 等，以免与外部预处理重复
- 合并推理表支持 --no-prob 忽略概率列
- bbox 裁剪可选：默认启用，传 --no-bbox 可关闭（不对 mask 做清洁）
"""
from __future__ import annotations
import argparse, logging, warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# ===== 静音 =====
try:
    import radiomics
    radiomics.setVerbosity(logging.ERROR)
    logging.getLogger("radiomics").setLevel(logging.ERROR)
except Exception:
    pass
warnings.filterwarnings("ignore", message="^Feature .* is deprecated")
warnings.filterwarnings("ignore", message="^GLCM is symmetrical.*")
warnings.filterwarnings("ignore", message=".*deprecated.*")

# ===== PyRadiomics =====
try:
    from radiomics.featureextractor import RadiomicsFeatureExtractor
except Exception as e:
    try:
        from radiomics import featureextractor as _fe  # type: ignore
        RadiomicsFeatureExtractor = _fe.RadiomicsFeatureExtractor  # type: ignore
    except Exception:
        raise ImportError(
            "未检测到正确的 PyRadiomics 安装。\n"
            "  pip uninstall -y radiomics\n"
            "  pip install -U pyradiomics SimpleITK PyWavelets\n"
            "注意：包名是 'pyradiomics'（不是 'radiomics'）"
        ) from e

# ===== SimpleITK 预处理 =====
import SimpleITK as sitk

TARGET_SPACING = (1.0, 1.0, 1.0)  # 统一重采样

# ---------- 列名标准化 ----------
def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    low = {c.lower(): c for c in df.columns}
    required = {
        "image_path": ["image_path", "imagepath", "img_path", "ct_path", "ct"],
        "mask_path":  ["mask_path",  "maskpath",  "seg_path", "label_path", "mask"],
    }
    rename: Dict[str, str] = {}
    for std, alts in required.items():
        hit = None
        for a in alts:
            if a in low:
                hit = low[a]; break
        if hit is None:
            raise ValueError(f"CSV 缺少列：{std}（允许别名：{alts}）")
        rename[hit] = std
    for a in ["case_id", "caseid", "case", "id", "编号"]:
        if a in df.columns:
            rename[a] = "case_id"; break
    return df.rename(columns=rename)

def default_case_id(image_path: Path) -> str:
    name = image_path.name
    if name.endswith(".nii.gz"): stem = name[:-7]
    elif name.endswith(".nii"): stem = name[:-4]
    else: stem = image_path.stem
    for suf in ["_image", "_img", "_ct"]:
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
    return stem

# ---------- 清洗特征 ----------
DEPRECATED_FEATURE_KEYS = {"Compactness1","Compactness2","SphericalDisproportion","StandardDeviation"}

def sanitize_features(feats: Dict, drop_deprecated: bool=True, prefer_glcm_joint_over_sum: bool=True) -> Dict:
    cleaned: Dict[str, object] = {}
    has_joint_avg, has_sum_avg = False, False
    for k, v in feats.items():
        k_str = str(k)
        if k_str.startswith("diagnostics_"):
            continue  # 永远不保留 diagnostics
        if drop_deprecated and any(k_str.endswith("_"+dep) or k_str==dep for dep in DEPRECATED_FEATURE_KEYS):
            continue
        if k_str.endswith("_glcm_JointAverage") or k_str=="JointAverage":
            has_joint_avg = True
        if k_str.endswith("_glcm_SumAverage") or k_str=="SumAverage":
            has_sum_avg = True
        cleaned[k_str] = v
    if prefer_glcm_joint_over_sum and has_joint_avg and has_sum_avg:
        for k in list(cleaned.keys()):
            if k.endswith("_glcm_SumAverage") or k=="SumAverage":
                cleaned.pop(k, None)
    return cleaned

# ---------- 推理表 & 合并 ----------
def _norm_path_str(p: str) -> str:
    try: return Path(p).resolve().as_posix().lower()
    except Exception: return str(p).replace("\\","/").lower()

def _basename(p: str) -> str:
    return Path(p).name.lower()

def _standardize_infer(df: pd.DataFrame, ignore_prob: bool=False) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    low = {c.lower(): c for c in df.columns}
    for std, alts in {
        "image_path": ["image_path","imagepath","img_path","ct_path","ct"],
        "mask_path":  ["mask_path","maskpath","seg_path","label_path","mask"],
        "case_id":    ["case_id","caseid","case","id","编号"],
    }.items():
        for a in alts:
            if a in low:
                df.rename(columns={low[a]: std}, inplace=True); break
    for c in ["gata6_label","label","gata6","pred"]:
        if c in low:
            src = low[c]
            if src != "gata6_label":
                df.rename(columns={src:"gata6_label"}, inplace=True)
            break
    if not ignore_prob:
        for c in ["gata6_prob","prob_pos","prob","p","score","probability","pred_prob"]:
            if c in low:
                src = low[c]
                if src != "gata6_prob":
                    df.rename(columns={src:"gata6_prob"}, inplace=True)
                break
    else:
        for c in ["gata6_prob","prob_pos","prob","p","score","probability","pred_prob"]:
            if c in df.columns:
                df.drop(columns=[c], inplace=True, errors="ignore")
    if "case_id" in df.columns:
        df["case_id"] = df["case_id"].astype(str).str.strip()
    if "gata6_label" in df.columns:
        df["gata6_label"] = pd.to_numeric(df["gata6_label"], errors="coerce").round().astype("Int64")
    if (not ignore_prob) and ("gata6_prob" in df.columns):
        df["gata6_prob"] = pd.to_numeric(df["gata6_prob"], errors="coerce")
    return df

def _majority_vote_int(s: pd.Series):
    v = pd.to_numeric(s, errors="coerce").dropna()
    if len(v)==0: return np.nan
    v = v.round().astype(int)
    return int(v.value_counts().idxmax())

def _dedup_by_caseid(df: pd.DataFrame, source: str) -> pd.DataFrame:
    d = df.copy()
    d["case_id"] = d["case_id"].astype(str).str.strip()
    if source=="infer":
        agg: Dict[str, Callable] = {}
        others = [c for c in d.columns if c not in ["case_id","gata6_prob","gata6_label"]]
        for c in others: agg[c] = "first"
        if "gata6_prob" in d.columns:
            d["gata6_prob"] = pd.to_numeric(d["gata6_prob"], errors="coerce"); agg["gata6_prob"] = "mean"
        if "gata6_label" in d.columns:
            d["gata6_label"] = pd.to_numeric(d["gata6_label"], errors="coerce"); agg["gata6_label"] = _majority_vote_int
        return d.groupby("case_id", as_index=False).agg(agg) if agg else d.drop_duplicates("case_id")
    return d.drop_duplicates(subset=["case_id"], keep="first")

def _try_merge(feats: pd.DataFrame, infer: pd.DataFrame, key: str="case_id", ignore_prob: bool=False
) -> Tuple[pd.DataFrame, float, str, pd.DataFrame, pd.DataFrame]:
    f = feats.copy(); g = infer.copy()
    if "image_path" in f.columns:
        f["_norm_path"] = f["image_path"].astype(str).map(_norm_path_str)
        f["_base"] = f["image_path"].astype(str).map(_basename)
    if "image_path" in g.columns:
        g["_norm_path"] = g["image_path"].astype(str).map(_norm_path_str)
        g["_base"] = g["image_path"].astype(str).map(_basename)
    if "case_id" in f.columns: f["case_id"] = f["case_id"].astype(str).str.strip()
    if "case_id" in g.columns: g["case_id"] = g["case_id"].astype(str).str.strip()

    def _merge_on(colname: str):
        if colname not in f.columns or colname not in g.columns: return None, 0.0
        ff, gg = f.copy(), g.copy()
        if colname=="case_id":
            ff = _dedup_by_caseid(ff, "feats")
            gg = _dedup_by_caseid(gg, "infer")
            keep = ["case_id"]
            if (not ignore_prob) and ("gata6_prob" in gg.columns): keep.append("gata6_prob")
            if "gata6_label" in gg.columns: keep.append("gata6_label")
            gg = gg[keep]
            merged = pd.merge(ff, gg, on="case_id", how="left")
        else:
            ff[colname] = ff[colname].astype(str); gg[colname] = gg[colname].astype(str)
            merged = pd.merge(ff, gg, on=colname, how="left", suffixes=("", "_inf"))
        hit_cols = []
        if (not ignore_prob) and ("gata6_prob" in merged.columns): hit_cols.append(merged["gata6_prob"].notna())
        if "gata6_label" in merged.columns: hit_cols.append(merged["gata6_label"].notna())
        hit = np.any(np.stack(hit_cols,1),1).sum() if hit_cols else 0
        return merged, float(hit)/max(1,len(ff))

    used="none"; merged=None; ratio=0.0
    if key in ("case_id","auto"):
        m3, r3 = _merge_on("case_id")
        if m3 is not None: merged, ratio, used = m3, r3, "case_id"
    if (key=="basename") or (key=="auto" and ratio<0.5):
        m2, r2 = _merge_on("_base")
        if (m2 is not None) and (r2>ratio): merged, ratio, used = m2, r2, "basename"
    if (key=="image_path") or (key=="auto" and ratio<0.5):
        m1, r1 = _merge_on("_norm_path")
        if (m1 is not None) and (r1>ratio): merged, ratio, used = m1, r1, "image_path"
    if merged is None: merged=f.copy(); ratio=0.0; used="none"

    if "gata6_label" in merged.columns:
        feats_unmatched = merged[merged["gata6_label"].isna()]
    elif (not ignore_prob) and ("gata6_prob" in merged.columns):
        feats_unmatched = merged[merged["gata6_prob"].isna()]
    else:
        feats_unmatched = merged.copy()

    infer_unmatched = pd.DataFrame()
    if used=="image_path":
        infer_unmatched = g[~g["_norm_path"].isin(merged.get("_norm_path", pd.Series([], dtype=str)))]
    elif used=="basename":
        infer_unmatched = g[~g["_base"].isin(merged.get("_base", pd.Series([], dtype=str)))]
    elif used=="case_id" and "case_id" in g.columns:
        infer_unmatched = g[~g["case_id"].isin(merged.get("case_id", pd.Series([], dtype=str)))]
    for col in ["_norm_path","_base"]:
        if col in merged.columns: merged.drop(columns=[col], inplace=True)
        if col in infer_unmatched.columns: infer_unmatched.drop(columns=[col], inplace=True)
    return merged, ratio, used, feats_unmatched, infer_unmatched

# ---------- 影像标准化：重采样 + [可选]bbox 裁剪 + ROI z-score ----------
def _to_uint8_mask(msk: sitk.Image) -> sitk.Image:
    if msk.GetPixelID()!=sitk.sitkUInt8:
        msk = sitk.Cast(msk, sitk.sitkUInt8)
    # 非零→1，保持人工标注拓扑，不做任何清洁
    return sitk.BinaryThreshold(msk, 1, 2**31-1, 1, 0)

def _resample(img: sitk.Image, spacing_out, is_mask: bool) -> sitk.Image:
    spacing_in = np.array(list(img.GetSpacing()), float)
    spacing_out = np.array(spacing_out, float)
    size_in = np.array(list(img.GetSize()), int)
    size_out = np.maximum(1, np.round(size_in * spacing_in / spacing_out)).astype(int)
    r = sitk.ResampleImageFilter()
    r.SetOutputSpacing(tuple(spacing_out.tolist()))
    r.SetSize([int(x) for x in size_out.tolist()])
    r.SetOutputDirection(img.GetDirection())
    r.SetOutputOrigin(img.GetOrigin())
    r.SetOutputPixelType(img.GetPixelID())
    r.SetTransform(sitk.Transform())
    r.SetDefaultPixelValue(0)
    r.SetInterpolator(sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear)
    return r.Execute(img)

def _crop_to_mask_bbox(img: sitk.Image, msk: sitk.Image, pad: Tuple[int,int,int]=(0,0,0)) -> Tuple[sitk.Image, sitk.Image]:
    """按 mask=1 的外接框裁剪；pad 为各向体素 padding；不做连通域筛选。"""
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(msk)
    labels = list(stats.GetLabels())
    if 1 not in labels:  # 兜底：无前景则不裁剪
        return img, msk
    x, y, z, sx, sy, sz = stats.GetBoundingBox(1)
    px, py, pz = pad
    x = max(0, x - px); y = max(0, y - py); z = max(0, z - pz)
    sx = min(int(img.GetSize()[0] - x), sx + 2*px)
    sy = min(int(img.GetSize()[1] - y), sy + 2*py)
    sz = min(int(img.GetSize()[2] - z), sz + 2*pz)
    size = [int(sx), int(sy), int(sz)]
    index = [int(x), int(y), int(z)]
    roi = sitk.RegionOfInterest
    return roi(img, size, index), roi(msk, size, index)

def _zscore_roi(img: sitk.Image, msk: sitk.Image) -> sitk.Image:
    a = sitk.GetArrayFromImage(img).astype(np.float32)
    m = sitk.GetArrayFromImage(msk) > 0
    roi = a[m] if m.any() else a
    mu = float(np.mean(roi)); sd = float(np.std(roi) + 1e-6)
    a = (a - mu) / sd
    out = sitk.GetImageFromArray(a); out.CopyInformation(img)
    return out

def preprocess_image_and_mask(img_path: Path, msk_path: Path, use_bbox: bool=True, bbox_pad_vox: int=0) -> Tuple[sitk.Image, sitk.Image]:
    """
    标准化：重采样1mm → 二值化 → [可选]按mask外接框裁剪(可padding) → ROI内z-score
    不对 mask 进行任何“清洁”操作（不取最大连通域、不填洞等）。
    """
    img = sitk.ReadImage(str(img_path))
    msk = sitk.ReadImage(str(msk_path))

    # 二值化（非零即前景）
    msk = _to_uint8_mask(msk)

    # 重采样
    img = _resample(img, TARGET_SPACING, is_mask=False)
    msk = _resample(msk, TARGET_SPACING, is_mask=True)
    msk = _to_uint8_mask(msk)

    # [可选] 裁剪到 bbox（可选 padding）
    if use_bbox:
        pad = (bbox_pad_vox, bbox_pad_vox, bbox_pad_vox)
        img, msk = _crop_to_mask_bbox(img, msk, pad=pad)

    # ROI 内 z-score
    img = _zscore_roi(img, msk)
    return img, msk

# ---------- 量化配置（binCount / binWidth） ----------
def _apply_quantization(extractor: RadiomicsFeatureExtractor, bin_count: Optional[int], bin_width: Optional[float]) -> Dict[str, object]:
    # 清理潜在冲突键
    for k in ["binWidth", "binCount"]:
        try:
            extractor.settings.pop(k, None)
        except Exception:
            pass
    meta = {}
    if bin_count is not None:
        extractor.settings["binCount"] = int(bin_count)
        meta["pre_bin_mode"] = "binCount"
        meta["pre_bin_param"] = int(bin_count)
    elif bin_width is not None:
        extractor.settings["binWidth"] = float(bin_width)
        meta["pre_bin_mode"] = "binWidth"
        meta["pre_bin_param"] = float(bin_width)
    else:
        extractor.settings["binCount"] = 64
        meta["pre_bin_mode"] = "binCount"
        meta["pre_bin_param"] = 64
    return meta

def _disable_internal_preproc(extractor: RadiomicsFeatureExtractor):
    extractor.settings["additionalInfo"] = False      # 不要 diagnostics
    extractor.settings["normalize"] = False
    extractor.settings["normalizeScale"] = 1
    extractor.settings["removeOutliers"] = None
    extractor.settings["resampledPixelSpacing"] = None
    extractor.settings["preCrop"] = False
    extractor.settings["padDistance"] = 0
    extractor.settings["correctMask"] = False
    extractor.settings["label"] = 1
    extractor.settings["interpolator"] = "sitkBSpline"

# ---------- 主流程 ----------
def extract_radiomics_from_csv(
    csv_path: Path,
    config_file: Path,
    output_csv: Path,
    inference_csv: Optional[Path] = None,
    join_key: str = "case_id",
    disable_cext: bool = False,
    drop_deprecated: bool = True,
    drop_glcm_sumaverage_if_joint: bool = True,
    no_prob: bool = False,
    keep_meta: bool = False,   # 是否保留 pre_* 元信息（默认不保留）
    use_bbox: bool = True,     # 是否使用 bbox 裁剪（默认启用；--no-bbox 可关闭）
    bbox_pad: int = 0,         # bbox padding（体素数）
    bin_count: Optional[int] = 64,
    bin_width: Optional[float] = None,
) -> None:
    df_in = pd.read_csv(csv_path, encoding="utf-8-sig")
    df = standardize_columns(df_in)

    extractor = RadiomicsFeatureExtractor(str(config_file))
    if disable_cext:
        extractor.settings["enableCExtensions"] = False

    _disable_internal_preproc(extractor)
    bin_meta = _apply_quantization(extractor, bin_count, bin_width)

    results: List[Dict] = []
    missing: List[Dict] = []
    failures: List[Dict] = []

    for r in tqdm(list(df.itertuples(index=False)), desc="提取组学特征（统一标准化）", unit="case", leave=True):
        image_path = Path(getattr(r, "image_path"))
        mask_path  = Path(getattr(r, "mask_path"))
        case_id    = getattr(r, "case_id", None)
        if case_id is None or (isinstance(case_id, float) and np.isnan(case_id)):
            case_id = default_case_id(image_path)

        if (not image_path.exists()) or (not mask_path.exists()):
            missing.append({"case_id": case_id, "image_path": str(image_path), "mask_path": str(mask_path)})
            continue

        try:
            img_pp, msk_pp = preprocess_image_and_mask(
                image_path, mask_path, use_bbox=use_bbox, bbox_pad_vox=bbox_pad
            )
            feats = extractor.execute(img_pp, msk_pp)
            feats = sanitize_features(
                feats,
                drop_deprecated=drop_deprecated,
                prefer_glcm_joint_over_sum=drop_glcm_sumaverage_if_joint
            )
            row = {"case_id": str(case_id).strip(),
                   "image_path": str(image_path),
                   "mask_path": str(mask_path)}
            if keep_meta:
                row.update({
                    "pre_spacing": "1,1,1",
                    "pre_use_bbox": int(bool(use_bbox)),
                    "pre_bbox_pad": int(bbox_pad if use_bbox else 0),
                    "pre_zscore_roi": 1,
                    "pre_bin_mode": bin_meta["pre_bin_mode"],
                    "pre_bin_param": bin_meta["pre_bin_param"],
                })
            results.append({**row, **feats})
        except Exception as e:
            failures.append({"case_id": str(case_id).strip(),
                             "image_path": str(image_path),
                             "mask_path": str(mask_path),
                             "error": str(e)})

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if results:
        df_feats = pd.DataFrame(results)
        df_feats["case_id"] = df_feats["case_id"].astype(str).str.strip()
        df_feats = _dedup_by_caseid(df_feats, source="feats")

        base_fixed = ["case_id","image_path","mask_path"]
        if keep_meta:
            base_fixed += ["pre_spacing","pre_use_bbox","pre_bbox_pad","pre_zscore_roi","pre_bin_mode","pre_bin_param"]
        others = [c for c in df_feats.columns if c not in base_fixed]
        df_feats = df_feats[base_fixed + others]

        drop_cols = [c for c in df_feats.columns if c.startswith("diagnostics_")]
        if not keep_meta:
            drop_cols += [c for c in df_feats.columns if c in
                          {"pre_spacing","pre_use_bbox","pre_bbox_pad","pre_zscore_roi","pre_bin_mode","pre_bin_param"}]
        if drop_cols:
            df_feats.drop(columns=drop_cols, inplace=True, errors="ignore")

        if inference_csv is not None and Path(inference_csv).exists():
            df_inf = pd.read_csv(inference_csv, encoding="utf-8-sig")
            df_inf = _standardize_infer(df_inf, ignore_prob=no_prob)

            merged, ratio, used, feats_unmatched, inf_unmatched = _try_merge(
                df_feats, df_inf, key=join_key, ignore_prob=no_prob
            )

            out_fixed = ["case_id","image_path","mask_path"]
            if "gata6_label" in merged.columns: out_fixed.append("gata6_label")
            if (not no_prob) and ("gata6_prob" in merged.columns): out_fixed.append("gata6_prob")

            # 如不保留 meta，则清除
            for c in ["pre_spacing","pre_use_bbox","pre_bbox_pad","pre_zscore_roi","pre_bin_mode","pre_bin_param"]:
                if c in merged.columns and not keep_meta:
                    merged.drop(columns=[c], inplace=True, errors="ignore")

            other_cols = [c for c in merged.columns if c not in out_fixed]
            merged = merged[out_fixed + other_cols]

            diag_cols = [c for c in merged.columns if c.startswith("diagnostics_")]
            if diag_cols:
                merged.drop(columns=diag_cols, inplace=True, errors="ignore")

            merged.to_csv(output_csv, index=False, encoding="utf-8-sig")
            print(f"[DONE] 特征+标签 已保存：{output_csv}")
            print(f"[MERGE] used={used}, match_ratio={ratio:.1%}, no_prob={no_prob}")
        else:
            df_feats.to_csv(output_csv, index=False, encoding="utf-8-sig")
            print(f"[DONE] 成功提取 {len(df_feats)} 条特征（未合并推理表），保存：{output_csv}")
    else:
        print("[DONE] 没有成功提取任何特征。")

    if missing:
        pd.DataFrame(missing).to_csv(output_csv.parent/"missing_cases.csv", index=False, encoding="utf-8-sig")
        print(f"[INFO] 缺失样本 {len(missing)}，记录：missing_cases.csv")
    if failures:
        pd.DataFrame(failures).to_csv(output_csv.parent/"failed_cases.csv", index=False, encoding="utf-8-sig")
        print(f"[INFO] 失败样本 {len(failures)}，记录：failed_cases.csv")

# ===== CLI =====
def main():
    ap = argparse.ArgumentParser(
        description="按 CSV 提取 CT 组学特征（重采样1mm + [可选]bbox 裁剪 + ROI z-score；不输出diagnostics，默认不输出pre_*）"
    )
    ap.add_argument("--csv", required=True, help="包含 imagepath/maskpath（可选 caseid/编号）的 CSV")
    ap.add_argument("--config", required=True, help="PyRadiomics 配置文件（YAML/JSON 均可）")
    ap.add_argument("--out", required=True, help="输出 CSV（特征+标签）路径")
    ap.add_argument("--inference", default=None, help="推理结果 CSV（可无）")
    ap.add_argument("--join-key", default="case_id", choices=["case_id","image_path","basename","auto"],
                    help="合并键；默认 case_id，auto 为 case_id→basename→image_path")
    ap.add_argument("--disable-cext", action="store_true", help="若遇 C 扩展兼容问题可关闭")
    ap.add_argument("--keep-deprecated", action="store_true", help="保留已弃用特征（默认丢弃）")
    ap.add_argument("--keep-glcm-sumaverage", action="store_true", help="保留 SumAverage（默认在有 JointAverage 时丢弃）")
    ap.add_argument("--no-prob", action="store_true", help="合并推理表时忽略所有概率列")
    ap.add_argument("--keep-meta", action="store_true", help="保留预处理元信息列 pre_*（默认不保留）")

    # bbox 开关（默认启用；--no-bbox 关闭），与 padding
    ap.add_argument("--no-bbox", action="store_true", help="关闭按 mask 外接框裁剪（默认开启）")
    ap.add_argument("--bbox-pad", type=int, default=0, help="bbox 四周的体素 padding（默认0，紧贴 ROI）")

    # 量化参数（互斥）：默认 binCount=64；若给了 --bin-width 则覆盖
    q = ap.add_mutually_exclusive_group()
    q.add_argument("--bin-count", type=int, default=64, help="固定灰度级数（推荐配合 z-score），默认 64")
    q.add_argument("--bin-width", type=float, default=None, help="灰度宽度（与 z-score 一起用时建议 0.2~0.5）")

    args = ap.parse_args()

    extract_radiomics_from_csv(
        csv_path=Path(args.csv),
        config_file=Path(args.config),
        output_csv=Path(args.out),
        inference_csv=Path(args.inference) if args.inference else None,
        join_key=args.join_key,
        disable_cext=args.disable_cext,
        drop_deprecated=(not args.keep_deprecated),
        drop_glcm_sumaverage_if_joint=(not args.keep_glcm_sumaverage),
        no_prob=args.no_prob,
        keep_meta=args.keep_meta,
        use_bbox=(not args.no_bbox),
        bbox_pad=args.bbox_pad,
        bin_count=args.bin_count if args.bin_width is None else None,
        bin_width=args.bin_width,
    )

if __name__ == "__main__":
    main()
