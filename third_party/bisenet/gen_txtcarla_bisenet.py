import os
import re

left_dir = 'left'
seg_dir = 'seg'
height_dir = 'height'
f_pix = 831.38
root = {"/baai-cwm-1/baai_cwm_ml/public_data/scenes/stereo/StereoRBHM/CarlaSpeedbumpsV4/town01/",
"/baai-cwm-1/baai_cwm_ml/public_data/scenes/stereo/StereoRBHM/CarlaSpeedbumpsV4/town02/",
"/baai-cwm-1/baai_cwm_ml/public_data/scenes/stereo/StereoRBHM/CarlaSpeedbumpsV4/town03/",
"/baai-cwm-1/baai_cwm_ml/public_data/scenes/stereo/StereoRBHM/CarlaSpeedbumpsV4/town04/",
"/baai-cwm-1/baai_cwm_ml/public_data/scenes/stereo/StereoRBHM/CarlaSpeedbumpsV4/town05/",
"/baai-cwm-1/baai_cwm_ml/public_data/scenes/stereo/StereoRBHM/CarlaSpeedbumpsV4/town06/",
"/baai-cwm-1/baai_cwm_ml/public_data/scenes/stereo/StereoRBHM/CarlaSpeedbumpsV4/town07/"
}
#root = {"/baai-cwm-1/baai_cwm_ml/public_data/scenes/stereo/StereoRBHM/CarlaSpeedbumpsV4/town10/"}
for root_dir in root:
    # root_dir = './town10HD'
    for root_dir1 in os.listdir(root_dir):
        print(root_dir1)
        match = re.search(r'h(\d+)_', root_dir1)
        height = round(int(match.group(1)) * 0.1, 2)
        with open('train_bisenet.txt', 'a') as file:
                left_list = os.listdir(  os.path.join(root_dir,root_dir1, left_dir)  )
                right_dir = 'baseline_200/right'
                disp_dir = 'baseline_200/disparity'
                baseline = 200
                for each in left_list:
                    base_name, _ = os.path.splitext(each)
                    content = '{} {} {} {} {} {} {} {}\n'.format(os.path.join(root_dir,root_dir1,left_dir, each), 
                                                            os.path.join(root_dir,root_dir1,right_dir, each), 
                                                            os.path.join(root_dir,root_dir1,disp_dir, base_name + ".npy"),
                                                            os.path.join(root_dir,root_dir1,seg_dir, base_name + ".png").replace("/baai-cwm-1/baai_cwm_ml/public_data/scenes/stereo/StereoRBHM/CarlaSpeedbumpsV4/","/baai-cwm-1/baai_cwm_ml/public_data/scenes/stereo/StereoRBHM/CarlaSpeedbumpsV4/CarlaSpeedbumpsV4_PostProcess/"),
                                                            height,baseline,f_pix,
                                                            os.path.join(root_dir.replace("/baai-cwm-1/baai_cwm_ml/public_data/scenes/stereo/StereoRBHM/CarlaSpeedbumpsV4/","/baai-cwm-1/baai_cwm_ml/public_data/scenes/stereo/StereoRBHM/CarlaSpeedbumpsV4/height/"),root_dir1,height_dir, base_name + ".npy")
                                                            )
                    file.write(content)
                            
                            
                right_dir = 'baseline_100/right'
                disp_dir = 'baseline_100/disparity'
                baseline = 100
                for each in left_list:
                    base_name, _ = os.path.splitext(each)
                    content = '{} {} {} {} {} {} {} {}\n'.format(os.path.join(root_dir,root_dir1,left_dir, each), 
                                                            os.path.join(root_dir,root_dir1,right_dir, each), 
                                                            os.path.join(root_dir,root_dir1,disp_dir, base_name + ".npy"),
                                                            os.path.join(root_dir,root_dir1,seg_dir, base_name + ".png").replace("/baai-cwm-1/baai_cwm_ml/public_data/scenes/stereo/StereoRBHM/CarlaSpeedbumpsV4/","/baai-cwm-1/baai_cwm_ml/public_data/scenes/stereo/StereoRBHM/CarlaSpeedbumpsV4/CarlaSpeedbumpsV4_PostProcess/"),
                                                            height,baseline,f_pix,
                                                            os.path.join(root_dir.replace("/baai-cwm-1/baai_cwm_ml/public_data/scenes/stereo/StereoRBHM/CarlaSpeedbumpsV4/","/baai-cwm-1/baai_cwm_ml/public_data/scenes/stereo/StereoRBHM/CarlaSpeedbumpsV4/height/"),root_dir1,height_dir, base_name + ".npy")
                                                            )
                    file.write(content)
                            
                            
                right_dir = 'baseline_054/right'
                disp_dir = 'baseline_054/disparity'
                baseline = 54
                for each in left_list:
                    base_name, _ = os.path.splitext(each)
                    content = '{} {} {} {} {} {} {} {}\n'.format(os.path.join(root_dir,root_dir1,left_dir, each), 
                                                            os.path.join(root_dir,root_dir1,right_dir, each), 
                                                            os.path.join(root_dir,root_dir1,disp_dir, base_name + ".npy"),
                                                            os.path.join(root_dir,root_dir1,seg_dir, base_name + ".png").replace("/baai-cwm-1/baai_cwm_ml/public_data/scenes/stereo/StereoRBHM/CarlaSpeedbumpsV4/","/baai-cwm-1/baai_cwm_ml/public_data/scenes/stereo/StereoRBHM/CarlaSpeedbumpsV4/CarlaSpeedbumpsV4_PostProcess/"),
                                                            height,baseline,f_pix,
                                                            os.path.join(root_dir.replace("/baai-cwm-1/baai_cwm_ml/public_data/scenes/stereo/StereoRBHM/CarlaSpeedbumpsV4/","/baai-cwm-1/baai_cwm_ml/public_data/scenes/stereo/StereoRBHM/CarlaSpeedbumpsV4/height/"),root_dir1,height_dir, base_name + ".npy")
                                                            )
                    file.write(content)
                  
                right_dir = 'baseline_010/right'
                disp_dir = 'baseline_010/disparity'
                baseline = 10
                for each in left_list:
                    base_name, _ = os.path.splitext(each)
                    content = '{} {} {} {} {} {} {} {}\n'.format(os.path.join(root_dir,root_dir1,left_dir, each), 
                                                            os.path.join(root_dir,root_dir1,right_dir, each), 
                                                            os.path.join(root_dir,root_dir1,disp_dir, base_name + ".npy"),
                                                            os.path.join(root_dir,root_dir1,seg_dir, base_name + ".png").replace("/baai-cwm-1/baai_cwm_ml/public_data/scenes/stereo/StereoRBHM/CarlaSpeedbumpsV4/","/baai-cwm-1/baai_cwm_ml/public_data/scenes/stereo/StereoRBHM/CarlaSpeedbumpsV4/CarlaSpeedbumpsV4_PostProcess/"),
                                                            height,baseline,f_pix,
                                                            os.path.join(root_dir.replace("/baai-cwm-1/baai_cwm_ml/public_data/scenes/stereo/StereoRBHM/CarlaSpeedbumpsV4/","/baai-cwm-1/baai_cwm_ml/public_data/scenes/stereo/StereoRBHM/CarlaSpeedbumpsV4/height/"),root_dir1,height_dir, base_name + ".npy")
                                                            )
                    file.write(content)
                  