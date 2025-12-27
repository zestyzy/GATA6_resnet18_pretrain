#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一 NIfTI 命名与导出（.nii / .nii.gz），兼容 Python 3.9，带 tqdm 进度条

Usage:
  # 预览（不复制）
  python file_rename.py /path/origin --out /path/unified
  # 复制+层级化输出（推荐）
  python file_rename.py /path/origin --out /path/unified --apply --hier --no_preview
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from tqdm import tqdm

# ---------------- 小工具 ----------------
def is_nii_or_gz(p: Path) -> bool:
    n = p.name.lower()
    return n.endswith(".nii") or n.endswith(".nii.gz")

def ext_of(p: Path) -> str:
    return ".nii.gz" if p.name.lower().endswith(".nii.gz") else ".nii"

def ensure_print(s: str, no_preview: bool):
    if not no_preview:
        print(s)

def path_parts_lower(p: Path) -> List[str]:
    return [x.lower() for x in p.parts]

# ---------------- inst1：同夹下“数字前缀”配对 ----------------
INST1_PAIR_PAT = re.compile(
    r"^(?P<id>\d+?)_(?P<role>image2|image|label)\.nii(?:\.gz)?$",
    re.IGNORECASE,
)

def collect_inst1(root: Path) -> List[Tuple[Path, str, str, str, str]]:
    """
    .../inst1/<case_dir>/01_image.nii, 01_label.nii, 02_image.nii ...
    导出为 out/inst1/<case_dir>/<ID>/{image,label,image2}.nii*
    """
    plan: List[Tuple[Path, str, str, str, str]] = []
    for base in root.rglob("*"):
        if not base.is_dir() or "88例来自机构1的无标签患者" not in base.name.lower():
            continue
        for case_dir in base.iterdir():
            if not case_dir.is_dir():
                continue
            caseid = case_dir.name
            groups: Dict[str, Dict[str, Path]] = defaultdict(dict)
            for f in case_dir.iterdir():
                if not f.is_file() or not is_nii_or_gz(f):
                    continue
                m = INST1_PAIR_PAT.match(f.name.lower())
                if not m:
                    continue
                gid = m.group("id")
                role = m.group("role")
                if role not in groups[gid]:
                    groups[gid][role] = f
            for gid, files in sorted(groups.items(), key=lambda kv: int(kv[0])):
                if "image" in files:
                    plan.append((files["image"], f"inst1/{caseid}/{gid}/image{ext_of(files['image'])}",
                                 "inst1", caseid, "image"))
                if "image2" in files:
                    plan.append((files["image2"], f"inst1/{caseid}/{gid}/image2{ext_of(files['image2'])}",
                                 "inst1", caseid, "image2"))
                if "label" in files:
                    plan.append((files["label"], f"inst1/{caseid}/{gid}/label{ext_of(files['label'])}",
                                 "inst1", caseid, "label"))
    return plan

# ---------------- 六院（编号目录下，只取 *_image / *_3D） ----------------
def collect_six(root: Path) -> List[Tuple[Path, str, str, str, str]]:
    plan: List[Tuple[Path, str, str, str, str]] = []
    for base in root.rglob("*"):
        if not base.is_dir() or ("六院" not in base.name):
            continue
        for case_dir in base.iterdir():
            if not case_dir.is_dir():
                continue
            caseid = case_dir.name
            img_file = None
            lab_file = None
            for f in case_dir.iterdir():
                if not f.is_file() or not is_nii_or_gz(f):
                    continue
                n = f.name.lower()
                if re.search(r"_image\.nii(\.gz)?$", n):
                    img_file = f
                elif re.search(r"_3d\.nii(\.gz)?$", n):
                    lab_file = f
            if img_file is not None:
                plan.append((img_file, f"six/{caseid}/image{ext_of(img_file)}", "six", caseid, "image"))
            if lab_file is not None:
                plan.append((lab_file, f"six/{caseid}/label{ext_of(lab_file)}", "six", caseid, "label"))
    return plan

# ---------------- 公共数据库（编号目录下，tumor→label, image→image） ----------------
def collect_public(root: Path) -> List[Tuple[Path, str, str, str, str]]:
    plan: List[Tuple[Path, str, str, str, str]] = []
    for base in root.rglob("*"):
        if not base.is_dir() or ("公共数据库" not in base.name):
            continue
        for case_dir in base.iterdir():
            if not case_dir.is_dir():
                continue
            caseid = case_dir.name
            img_file = None
            lab_file = None
            for f in case_dir.iterdir():
                if not f.is_file() or not is_nii_or_gz(f):
                    continue
                n = f.name.lower()
                if ("tumor" in n) or ("tumour" in n):
                    lab_file = f
                elif "image" in n or "img" in n:
                    img_file = f
            if img_file is not None:
                plan.append((img_file, f"public/{caseid}/image{ext_of(img_file)}", "public", caseid, "image"))
            if lab_file is not None:
                plan.append((lab_file, f"public/{caseid}/label{ext_of(lab_file)}", "public", caseid, "label"))
    return plan

# ---------------- 中山医院有标签（病例目录下，优先选择 image / label） ----------------
# label 选择优先级：new_mask > label > mask > tumor/seg
ZS_LABEL_PATS = [
    re.compile(r"new_mask\.nii(\.gz)?$", re.I),
    re.compile(r"label\.nii(\.gz)?$", re.I),
    re.compile(r"mask\.nii(\.gz)?$", re.I),
    re.compile(r"(tumou?r|seg)\.nii(\.gz)?$", re.I),
]
def collect_zs(root: Path) -> List[Tuple[Path, str, str, str, str]]:
    plan: List[Tuple[Path, str, str, str, str]] = []
    for base in root.rglob("*"):
        if not base.is_dir() or ("中山医院有标签" not in base.name):
            continue
        for case_dir in base.iterdir():
            if not case_dir.is_dir():
                continue
            caseid = case_dir.name
            img_file = None
            label_candidates: List[Path] = []
            for f in case_dir.iterdir():
                if not f.is_file() or not is_nii_or_gz(f):
                    continue
                n = f.name.lower()
                if ("image" in n) or ("img" in n):
                    img_file = img_file or f
                for pat in ZS_LABEL_PATS:
                    if pat.search(n):
                        label_candidates.append(f)
                        break
            # 选优 label
            lab_file = None
            for pat in ZS_LABEL_PATS:
                chosen = next((p for p in label_candidates if pat.search(p.name.lower())), None)
                if chosen is not None:
                    lab_file = chosen
                    break
            if img_file is not None:
                plan.append((img_file, f"zs/{caseid}/image{ext_of(img_file)}", "zs", caseid, "image"))
            if lab_file is not None:
                plan.append((lab_file, f"zs/{caseid}/label{ext_of(lab_file)}", "zs", caseid, "label"))
    return plan

# ---------------- MSD（完整移植你的配对规则） ----------------
def is_nii_gz_full(path: Path) -> bool:
    return path.suffixes[-2:] == [".nii", ".gz"]

def get_stem(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return path.stem

def msd_split_and_role(path: Path) -> Tuple[Optional[str], Optional[str]]:
    parts = [p.name.lower() for p in path.parents]
    split = role = None
    if "imagestr" in parts:
        split, role = "tr", "image"
    elif "imagests" in parts:
        split, role = "ts", "image"
    elif "labelstr" in parts:
        split, role = "tr", "label"
    elif "labelsts" in parts:
        split, role = "ts", "label"
    return split, role

def under_msd_root(path: Path) -> bool:
    return any(p.name.lower() == "msd" for p in path.parents)

MSD_NUM_PATS = [
    re.compile(r"^pancreas[_-]?(\d{1,4})$", re.I),
    re.compile(r"^(\d{1,4})$"),
]
def extract_msd_numeric_id(stem: str) -> Optional[int]:
    for pat in MSD_NUM_PATS:
        m = pat.match(stem)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                pass
    return None

def collect_msd(root: Path, no_preview: bool) -> Tuple[List[Tuple[Path, str, str, str, str]], int]:
    """
    返回 (plan, missing_pairs)
    导出到: msd/tr/<caseid>/{image,label}.nii* 或 msd/ts/<caseid>/{image,label}.nii*
    """
    plan: List[Tuple[Path, str, str, str, str]] = []
    missing_pairs = 0

    nii_list: List[Path] = []
    for pattern in ("*.nii", "*.nii.gz"):
        nii_list.extend(list(root.rglob(pattern)))
    if not nii_list:
        return plan, missing_pairs

    msd_tr_images: Dict[str, Path] = {}
    msd_tr_labels: Dict[str, Path] = {}
    msd_ts_images_stem: Dict[str, Path] = {}
    msd_ts_labels_stem: Dict[str, Path] = {}
    msd_ts_images_id: Dict[int, Path] = {}
    msd_ts_labels_id: Dict[int, Path] = {}

    non_msd_files: List[Path] = []

    for src in tqdm(sorted(nii_list), desc="扫描 MSD", unit="file"):
        split, mrole = msd_split_and_role(src)
        if split is None or not under_msd_root(src):
            non_msd_files.append(src)
            continue
        stem = get_stem(src)
        if split == "tr":
            if mrole == "image":
                msd_tr_images[stem] = src
            else:
                msd_tr_labels[stem] = src
        else:  # ts
            if mrole == "image":
                msd_ts_images_stem[stem] = src
                id_ = extract_msd_numeric_id(stem)
                if id_ is not None and id_ not in msd_ts_images_id:
                    msd_ts_images_id[id_] = src
            else:
                msd_ts_labels_stem[stem] = src
                id_ = extract_msd_numeric_id(stem)
                if id_ is not None and id_ not in msd_ts_labels_id:
                    msd_ts_labels_id[id_] = src

    # TR 严格配对
    all_tr_stems = sorted(set(msd_tr_images) | set(msd_tr_labels))
    for stem in all_tr_stems:
        img = msd_tr_images.get(stem)
        lab = msd_tr_labels.get(stem)
        if img and lab:
            caseid = stem
            for role, src in (("image", img), ("label", lab)):
                rel = f"msd/tr/{caseid}/{role}{ext_of(src)}"
                plan.append((src, rel, "msd", caseid, role))
        else:
            missing_pairs += 1
            if not no_preview:
                miss = "缺 image" if (not img and lab) else ("缺 label" if (img and not lab) else "都缺?")
                print(f"[MSD未配对] TR {stem}: {miss}")

    # TS 先按完整 stem，再按数值 ID
    paired_ts_ids = set()

    exact_stems = sorted(set(msd_ts_images_stem) & set(msd_ts_labels_stem))
    for stem in exact_stems:
        img = msd_ts_images_stem[stem]
        lab = msd_ts_labels_stem[stem]
        id_ = extract_msd_numeric_id(stem)
        caseid = f"pancreas_{id_:03d}" if id_ is not None else stem
        for role, src in (("image", img), ("label", lab)):
            rel = f"msd/ts/{caseid}/{role}{ext_of(src)}"
            plan.append((src, rel, "msd", caseid, role))
        if id_ is not None:
            paired_ts_ids.add(id_)

    all_ids = sorted(set(msd_ts_images_id) | set(msd_ts_labels_id))
    for id_ in all_ids:
        if id_ in paired_ts_ids:
            continue
        img = msd_ts_images_id.get(id_)
        lab = msd_ts_labels_id.get(id_)
        if img and lab:
            caseid = f"pancreas_{id_:03d}"
            for role, src in (("image", img), ("label", lab)):
                rel = f"msd/ts/{caseid}/{role}{ext_of(src)}"
                plan.append((src, rel, "msd", caseid, role))
            paired_ts_ids.add(id_)
        else:
            missing_pairs += 1
            if not no_preview:
                miss = "缺 image" if (not img and lab) else ("缺 label" if (img and not lab) else "都缺?")
                print(f"[MSD未配对] TS id={id_:03d}: {miss}")

    return plan, missing_pairs

# ---------------- 通用兜底（跳过已覆盖集合） ----------------
LABEL_PAT = re.compile(r"(?:\blabel\b|\bmask\b|\btumou?r\b|\bseg\b|_3d\b)", re.I)
IMAGE_PAT = re.compile(r"\b(image|img)\b", re.I)
DATASET_ABBR = [
    (re.compile("六院", re.I), "six"),
    (re.compile("中山医院有标签|中山|zs", re.I), "zs"),
    (re.compile("公共数据库", re.I), "public"),
    (re.compile(r"\binst1\b|机构|无标签", re.I), "inst1"),
    (re.compile(r"\bmsd\b", re.I), "msd"),
]
def infer_dataset_generic(p: Path) -> str:
    for q, abbr in DATASET_ABBR:
        if any(q.search(x) for x in path_parts_lower(p)):
            return abbr
    return "misc"

def infer_caseid_generic(p: Path) -> str:
    return p.parent.name

def collect_generic(root: Path, exclude_targets: set[str]) -> List[Tuple[Path, str, str, str, str]]:
    plan: List[Tuple[Path, str, str, str, str]] = []
    for f in root.rglob("*"):
        if not (f.is_file() and is_nii_or_gz(f)):
            continue
        ds = infer_dataset_generic(f)
        if ds in {"six", "public", "inst1", "zs", "msd"}:
            continue
        role = "label" if LABEL_PAT.search(f.name) else ("image" if IMAGE_PAT.search(f.name) else "image")
        caseid = infer_caseid_generic(f)
        rel = f"{ds}/{caseid}/{role}{ext_of(f)}"
        if rel in exclude_targets:
            continue
        plan.append((f, rel, ds, caseid, role))
    return plan

# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, help="原始数据根目录")
    ap.add_argument("--out", type=Path, default=Path("./unified"), help="输出目录")
    ap.add_argument("--apply", action="store_true", help="执行复制（默认只预览）")
    ap.add_argument("--hier", action="store_true",
                    help="层级化输出：<out>/<dataset>/<caseid>/<type>.<ext>（建议开启）")
    ap.add_argument("--no_preview", action="store_true", help="预览时不打印明细（仅显示统计）")
    args = ap.parse_args()

    root = args.root

    # 1) 高优先级数据源
    plan_all: List[Tuple[Path, str, str, str, str]] = []
    plan_all += collect_inst1(root)
    plan_all += collect_six(root)
    plan_all += collect_public(root)
    plan_all += collect_zs(root)
    msd_plan, msd_missing = collect_msd(root, no_preview=args.no_preview)
    plan_all += msd_plan

    # 2) 其它兜底（避免覆盖）
    exclude_targets = set(dst for _, dst, *_ in plan_all)
    plan_all += collect_generic(root, exclude_targets)

    if not plan_all:
        print("未找到可处理的 .nii / .nii.gz 文件。")
        sys.exit(1)

    # 3) 目标路径构造 + 冲突消解（保留首次）
    final_plan: List[Tuple[Path, Path, str, str, str]] = []
    seen_dst = set()
    for src, rel, ds, cid, role in plan_all:
        dst = (args.out / rel) if args.hier else (args.out / rel.replace("/", "_"))
        key = dst.as_posix()
        if key in seen_dst:
            continue
        seen_dst.add(key)
        final_plan.append((src, dst, ds, cid, role))

    # 4) 预览或复制
    if not args.apply:
        ensure_print("===== 映射预览 =====", args.no_preview)
        by_ds = defaultdict(int)
        for src, dst, ds, cid, role in final_plan:
            by_ds[ds] += 1
            ensure_print(f"{src}  ->  {dst}   ({ds}/{cid}/{role})", args.no_preview)
        print("\n统计：")
        for ds, n in sorted(by_ds.items()):
            print(f"  {ds:>6s}: {n:5d} 个文件")
        if msd_missing:
            print(f"\n[提示] MSD 有 {msd_missing} 个 case 未成对（images*/labels* 不完整或无法匹配）。")
        print("\nDry-run 完成。确认无误后加 --apply 执行复制。")
        return

    copied = 0
    for src, dst, *_ in tqdm(final_plan, desc="复制文件", unit="file"):
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
    print(f"\n已复制 {copied} 个文件到 {args.out}")
    if msd_missing:
        print(f"[注意] 仍有 {msd_missing} 个 MSD case 因未配对而未导出。")

if __name__ == "__main__":
    main()
