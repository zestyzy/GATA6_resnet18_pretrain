import os
import pandas as pd

# === 输入输出路径 ===
csv_path = "/mnt/e/public/yzeng/codespace/yzeng/6yuan_task/task3_GATA6/data/csv/features/ssl_features/ssl_136_futures.csv"
excel_path = "/mnt/e/public/yzeng/codespace/yzeng/6yuan_task/task3_GATA6/data/csv/label_csv/GATA6_label.xlsx"
out_csv_path = "/mnt/e/public/yzeng/codespace/yzeng/6yuan_task/task3_GATA6/data/csv/features/ssl_features/ssl_136_futures_fixed.csv"

# === 读取 CSV 和 Excel ===
df = pd.read_csv(csv_path, dtype=str)
df_excel = pd.read_excel(excel_path, dtype=str)

# === 1. 从父目录名提取前缀编号（遇到 01芮芬 -> 01） ===
def get_parent_num(path):
    parent = os.path.basename(os.path.dirname(path))
    # 只保留开头的连续数字
    num = ''.join([ch for ch in parent if ch.isdigit()])
    return num

df['case_id'] = df['image_path'].apply(get_parent_num)

# === 2. Excel case_id 处理：去空格 + 补零两位 ===
df_excel['case_id'] = df_excel['case_id'].str.strip().str.zfill(2)
df['case_id'] = df['case_id'].str.strip().str.zfill(2)

# === 3. 合并 GATA6 信息 ===
df_merged = df.merge(df_excel, on="case_id", how="left")

# === 4. 按 case_id 数值升序排序 ===
df_merged = df_merged.sort_values(by="case_id", key=lambda x: x.astype(int))

# === 5. 保存结果 ===
df_merged.to_csv(out_csv_path, index=False, encoding="utf-8-sig")

print(f"[OK] 已修正 case_id 并合并 GATA6，按升序保存到: {out_csv_path}")
