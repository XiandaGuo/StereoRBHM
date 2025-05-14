import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import os
import tqdm


def recreate_directory_structure(src_dir, dst_dir):
    """
    复现源文件夹的目录结构到目标文件夹，只创建空文件夹
    """
    dirs_to_create = []

    for root, dirs, files in os.walk(src_dir):
        # 计算目标文件夹中的对应路径
        rel_path = os.path.relpath(root, src_dir)
        target_dir = os.path.join(dst_dir, rel_path)

        dirs_to_create.append(target_dir)

    # 批量创建目录
    for target_dir in dirs_to_create:
        try:
            os.makedirs(target_dir, exist_ok=True)
            print(f"Created directory: {target_dir}")
        except Exception as e:
            print(f"Error creating directory {target_dir}: {e}")

def process_image_single(input_path, output_path):
    # 读取图像
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    # 先膨胀后腐蚀（闭运算）
    kernel = np.ones((5, 5), np.uint8)  # 核大小可以调节
    img_dilated = cv2.dilate(img, kernel, iterations=2)  # 膨胀
    img_closed = cv2.erode(img_dilated, kernel, iterations=2)  # 腐蚀

    # 连通域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_closed, connectivity=8)

    # 找到面积最大的255区域
    max_label = 1  # 忽略背景0的标签
    max_area = stats[1, cv2.CC_STAT_AREA]
    for i in range(2, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > max_area:
            max_area = stats[i, cv2.CC_STAT_AREA]
            max_label = i

    # 创建新图像，只保留面积最大的255区域
    largest_region = np.zeros_like(img)
    largest_region[labels == max_label] = 255

    # 边缘平滑处理
    smoothed_img = cv2.GaussianBlur(largest_region, (5, 5), 0)

    # 保存最终结果
    cv2.imwrite(output_path, smoothed_img)

def process_image(input_path):

    output_path = input_path.replace('CarlaSpeedbumpsV4', 'CarlaSpeedbumpsV4_PostProcess')
    # 读取图像
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    # 先膨胀后腐蚀（闭运算）
    kernel = np.ones((5, 5), np.uint8)  # 核大小可以调节
    img_dilated = cv2.dilate(img, kernel, iterations=2)  # 膨胀
    img_closed = cv2.erode(img_dilated, kernel, iterations=2)  # 腐蚀


    # 连通域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_closed, connectivity=8)

    # 确保存在前景区域
    if num_labels > 1:
        # 找到面积最大的255区域
        max_label = 1  # 忽略背景0的标签
        max_area = stats[1, cv2.CC_STAT_AREA]
        for i in range(2, num_labels):
            if stats[i, cv2.CC_STAT_AREA] > max_area:
                max_area = stats[i, cv2.CC_STAT_AREA]
                max_label = i

        # 创建新图像，只保留面积最大的255区域
        largest_region = np.zeros_like(img)
        largest_region[labels == max_label] = 255

        # 边缘平滑处理
        smoothed_img = cv2.GaussianBlur(largest_region, (5, 5), 0)
    else:
        # 如果没有前景区域，直接返回空白图像
        print("no bump:", input_path)
        smoothed_img = np.zeros_like(img)

    # 保存最终结果
    cv2.imwrite(output_path, smoothed_img)


# 示例调用
# input_path = "/vepfs/dujun.nie/data/CarlaSpeedbumps_Segmentation/town03/h51_w34_foggy/seg/000742.png"#"/vepfs/dujun.nie/data/CarlaSpeedbumps_Segmentation/town10/h74_w21_foggy/seg/000671.png"  # 输入图片路径#
# output_path = '/vepfs/dujun.nie/debug2.png'  # 输出图片路径
# process_image_single(input_path, output_path)



recreate_directory_structure("/baai-cwm-1/baai_cwm_ml/cwm/xianda.guo/code/Wrl/speedbump/bisenet/CarlaSpeedbumpsV4/", '/baai-cwm-1/baai_cwm_ml/cwm/xianda.guo/code/Wrl/speedbump/bisenet/CarlaSpeedbumpsV4_PostProcess/')
all_img_path_file = "/baai-cwm-1/baai_cwm_ml/cwm/xianda.guo/code/Wrl/speedbump/v4/trainval_new.txt"

with open(all_img_path_file, 'r') as f:
    paths = sorted([line.strip().split(' ')[0].replace('/baai-cwm-1/baai_cwm_ml/public_data/scenes/stereo/StereoRBHM/CarlaSpeedbumpsV4/', '/baai-cwm-1/baai_cwm_ml/cwm/xianda.guo/code/Wrl/speedbump/bisenet/CarlaSpeedbumpsV4/').replace('/left/', '/seg/').replace('.jpg', '.png') for line in f.readlines()])
paths = list(set(paths))
with ThreadPoolExecutor() as executor:
    results = list(tqdm.tqdm(executor.map(process_image, paths), total=len(paths)))

