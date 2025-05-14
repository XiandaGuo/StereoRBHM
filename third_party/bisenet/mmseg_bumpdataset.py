import os
import numpy as np
from PIL import Image
import shutil
from tqdm import tqdm  # 导入 tqdm

# 定义文件路径
train_img_path_file = '/baai-cwm-1/baai_cwm_ml/cwm/xianda.guo/code/Wrl/speedbump/bisenet/origin_dataset/train_img.txt'
train_mask_path_file = '/baai-cwm-1/baai_cwm_ml/cwm/xianda.guo/code/Wrl/speedbump/bisenet/origin_dataset/train_mask.txt'
val_img_path_file = '/baai-cwm-1/baai_cwm_ml/cwm/xianda.guo/code/Wrl/speedbump/bisenet/origin_dataset/val_img.txt'
val_mask_path_file = '/baai-cwm-1/baai_cwm_ml/cwm/xianda.guo/code/Wrl/speedbump/bisenet/origin_dataset/val_mask.txt'

# 新数据集文件夹
dataset_dir = '/baai-cwm-nas/algorithm/xianda.guo/checkpoints/wrl/data/CarlaSpeedbumpv3_4finetune'
img_dir = os.path.join(dataset_dir, 'img_dir')
ann_dir = os.path.join(dataset_dir, 'ann_dir')

# 创建目录结构
os.makedirs(os.path.join(img_dir, 'train'), exist_ok=True)
os.makedirs(os.path.join(img_dir, 'val'), exist_ok=True)
os.makedirs(os.path.join(ann_dir, 'train'), exist_ok=True)
os.makedirs(os.path.join(ann_dir, 'val'), exist_ok=True)

# 定义调色板：背景（0类）和目标（1类）的颜色
palette = [0, 0, 0, 128, 64, 128]  # 前景（目标类）为[128, 64, 128]，背景类为[0, 0, 0]


def load_paths(file_path):
    """从txt文件中读取路径"""
    with open(file_path, 'r') as file:
        paths = [line.strip() for line in file]
    return paths


def copy_image_with_unique_name(img_path, dest_dir):
    """复制图像文件并生成唯一的文件名"""
    unique_name = generate_unique_filename(img_path, '.jpg')
    dest_path = os.path.join(dest_dir, unique_name)
    shutil.copy(img_path, dest_path)


def process_and_save_mask(mask_path, save_path):
    """处理并保存分割标注文件"""
    # 打开分割标注文件并转换为数组
    mask = np.array(Image.open(mask_path))
    # 将灰度值 230 转换为 1，其余为 0
    mask = np.where(mask == 230, 1, 0).astype(np.uint8)

    # 转换为调色板图像
    palette_image = Image.fromarray(mask, mode='P')
    palette_image.putpalette(palette)
    # 保存调色板图像
    palette_image.save(save_path)


def generate_unique_filename(original_path, ext):
    """根据原路径生成唯一的文件名"""
    # 提取路径中的 town 和其他唯一信息，确保文件名不重复
    parts = original_path.split(os.sep)
    town_info = parts[-4]  # 例如：town01
    scene_info = parts[-3]  # 例如：h38_w22_rainy

    # 获取文件名部分，去除扩展名
    base_name = os.path.splitext(parts[-1])[0]

    # 组合唯一的文件名
    unique_name = f"{town_info}_{scene_info}_{base_name}{ext}"

    return unique_name


# 读取图片和分割标注路径
train_img_paths = load_paths(train_img_path_file)
train_mask_paths = load_paths(train_mask_path_file)
val_img_paths = load_paths(val_img_path_file)
val_mask_paths = load_paths(val_mask_path_file)

# 复制训练集和验证集图片到目标文件夹，并显示进度
print("正在复制训练集图片...")
for img_path in tqdm(train_img_paths, desc="复制训练集图片", unit="个"):
    copy_image_with_unique_name(img_path, os.path.join(img_dir, 'train'))

print("正在复制验证集图片...")
for img_path in tqdm(val_img_paths, desc="复制验证集图片", unit="个"):
    copy_image_with_unique_name(img_path, os.path.join(img_dir, 'val'))

# 处理并保存训练集和验证集分割标注文件，并显示进度
print("正在处理训练集分割标注...")
for mask_path in tqdm(train_mask_paths, desc="处理训练集分割标注", unit="个"):
    # 生成唯一的文件名
    mask_name = generate_unique_filename(mask_path, '.png')
    save_path = os.path.join(ann_dir, 'train', mask_name)
    process_and_save_mask(mask_path, save_path)

print("正在处理验证集分割标注...")
for mask_path in tqdm(val_mask_paths, desc="处理验证集分割标注", unit="个"):
    # 生成唯一的文件名
    mask_name = generate_unique_filename(mask_path, '.png')
    save_path = os.path.join(ann_dir, 'val', mask_name)
    process_and_save_mask(mask_path, save_path)

print("数据集构建完成！")
