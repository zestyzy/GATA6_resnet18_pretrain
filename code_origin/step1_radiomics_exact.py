import os
import argparse
import logging
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm
from radiomics import featureextractor

# 关闭 PyRadiomics 的冗余日志（比如 GLCM 提示）
logging.getLogger("radiomics").setLevel(logging.ERROR)

def find_mask(img_path: Path):
    """
    根据 image 文件名找到对应 mask
    规则：替换 _image.nii / _image.nii.gz 为 _mask.nii / _mask.nii.gz
    """
    name = img_path.name
    if name.endswith("_image.nii.gz"):
        return img_path.parent / name.replace("_image.nii.gz", "_3D.nii.gz")
    elif name.endswith("_image.nii"):
        return img_path.parent / name.replace("_image.nii", "_label.nii")
    else:
        return None

def extract_radiomics_features(root_dir: Path, config_file: Path, output_csv: Path):
    """
    遍历 root_dir 下所有 *_image.nii(.gz)，根据 config_file 提取 radiomics 特征
    """
    extractor = featureextractor.RadiomicsFeatureExtractor(str(config_file))
    # 如果遇到 numpy ABI 冲突，可以加上这一行：
    # extractor.settings['enableCExtensions'] = False

    image_files = list(root_dir.rglob("*_image.nii")) + list(root_dir.rglob("*_image.nii.gz"))
    if not image_files:
        print(f"[WARN] 在 {root_dir} 下没有找到 *_image.nii 或 *_image.nii.gz 文件。")
        return

    results, failures, missing = [], [], []

    for img_path in tqdm(sorted(image_files), desc="提取组学特征", unit="case"):
        mask_path = find_mask(img_path)
        if mask_path is None or not mask_path.exists():
            missing.append({"image": str(img_path), "mask": str(mask_path)})
            continue

        try:
            feature_vector = extractor.execute(str(img_path), str(mask_path))
            row = {
                "case_id": img_path.stem.replace("_image", ""),
                "image_path": str(img_path),
                "mask_path": str(mask_path),
            }
            # 去掉 diagnostics 信息
            for k, v in feature_vector.items():
                if not str(k).startswith("diagnostics_"):
                    row[k] = v
            results.append(row)
        except Exception as e:
            failures.append({"image": str(img_path), "mask": str(mask_path), "error": str(e)})

    # 保存结果
    output_dir = output_csv.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    if results:
        df = pd.DataFrame(results)
        fixed_cols = ["case_id", "image_path", "mask_path"]
        other_cols = [c for c in df.columns if c not in fixed_cols]
        df = df[fixed_cols + other_cols]
        df.to_csv(output_csv, index=False, encoding="utf-8-sig")
        print(f"[DONE] 成功提取 {len(df)} 条特征，已保存到 {output_csv}")
    else:
        print("[DONE] 没有成功提取到任何特征。")

    # 保存缺失和失败记录
    if missing:
        pd.DataFrame(missing).to_csv(output_dir / "missing_cases.csv", index=False)
        print(f"[INFO] 有 {len(missing)} 个缺失掩码，已记录到 missing_cases.csv")
    if failures:
        pd.DataFrame(failures).to_csv(output_dir / "failed_cases.csv", index=False)
        print(f"[INFO] 有 {len(failures)} 个失败样本，已记录到 failed_cases.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量CT影像组学特征提取")
    parser.add_argument("--root", type=str, required=True, help="包含 *_image.nii(.gz) 文件的目录")
    parser.add_argument("--config", type=str, required=True, help="Radiomics 配置文件路径 (JSON)")
    parser.add_argument("--out", type=str, required=True, help="输出 CSV 文件路径")
    args = parser.parse_args()

    extract_radiomics_features(Path(args.root), Path(args.config), Path(args.out))
