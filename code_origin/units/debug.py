import os
import nibabel as nib
import numpy as np
from tqdm import tqdm   # 进度条

# 1. 根目录，按需修改
root_dir = '/mnt/e/public/yzeng/codespace/yzeng/6yuan_task/task3_GATA6/data/pipline/ssl/Step2_make_persudo/dataset/dcm/msd/tr'

# 2. 先收集所有待处理文件路径
label_files = []
for dirpath, _, files in os.walk(root_dir):
    if 'label.nii.gz' in files:
        label_files.append(os.path.join(dirpath, 'label.nii.gz'))

print(f'Found {len(label_files)} label.nii.gz files')

# 3. 批量处理并显示进度条
for f_path in tqdm(label_files, desc='Remapping labels'):
    img = nib.load(f_path)
    data = img.get_fdata().astype(np.uint8)

    data_new = data.copy()
    data_new[data == 1] = 0
    data_new[data == 2] = 1

    new_img = nib.Nifti1Image(data_new, img.affine, img.header)
    nib.save(new_img, f_path)

print('All done!')