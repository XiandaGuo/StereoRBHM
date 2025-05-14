from mmseg.apis import init_model, inference_model, show_result_pyplot
import os
import cv2
import tqdm
from concurrent.futures import ThreadPoolExecutor
import numpy as np

config_path = "/baai-cwm-1/baai_cwm_ml/cwm/xianda.guo/code/Wrl/speedbump/bisenet/mmsegmentation-main/work_dirs/bisenetv2_fcn_4xb4-160k_speedbump-1024x1024/bisenetv2_fcn_4xb4-160k_speedbump-1024x1024.py"
checkpoint_path = "/baai-cwm-1/baai_cwm_ml/cwm/xianda.guo/code/Wrl/speedbump/bisenet/mmsegmentation-main/work_dirs/bisenetv2_fcn_4xb4-160k_speedbump-1024x1024/iter_40000.pth"
model = init_model(config_path, checkpoint_path, device='cuda:0')

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

def process_one_image(img_path):
    # inference on given image
    print(img_path)
    save_path = img_path.replace('/baai-cwm-1/baai_cwm_ml/public_data/scenes/stereo/StereoRBHM/CarlaSpeedbumpsV4/', "/baai-cwm-1/baai_cwm_ml/cwm/xianda.guo/code/Wrl/speedbump/bisenet/CarlaSpeedbumpsV4/").replace('/left/', '/seg/').replace('.jpg', '.png')
    result = inference_model(model, img_path)
    seg_map = (result.pred_sem_seg.data.cpu().numpy().squeeze() * 255).astype(np.uint8)  # 提取分割图并转换为 numpy 数组
    cv2.imwrite(save_path, seg_map)

recreate_directory_structure('/baai-cwm-1/baai_cwm_ml/public_data/scenes/stereo/StereoRBHM/CarlaSpeedbumpsV4/', '/baai-cwm-1/baai_cwm_ml/cwm/xianda.guo/code/Wrl/speedbump/bisenet/CarlaSpeedbumpsV4/')
train_img_path_file = "/baai-cwm-1/baai_cwm_ml/cwm/xianda.guo/code/Wrl/speedbump/v4/trainval_new.txt"
#val_img_path_file = '/vepfs/dujun.nie/val_img_path.txt'

with open(train_img_path_file, 'r') as f:
    paths = sorted([line.strip().split(' ')[0] for line in f.readlines()])


paths = list(set(paths))
print(len(paths))
#with open(val_img_path_file, 'r') as f:
    #val_paths = sorted([line.strip() for line in f.readlines()])

#paths = train_paths + val_paths

with ThreadPoolExecutor() as executor:
    results = list(tqdm.tqdm(executor.map(process_one_image, paths), total=len(paths)))