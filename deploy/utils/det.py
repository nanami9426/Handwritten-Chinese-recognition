import os
import subprocess
import re
WORKING_DIR = './'
PIC_FOLDER = '../dist/uploads/'

def det(picname):
    os.chdir(WORKING_DIR)
    picsrc = PIC_FOLDER + picname
    command = ['python','tools/infer/predict_det.py',
               f'--image_dir={picsrc}',
               f'--det_model_dir=./ch_PP-OCRv3_det_infer',
               f'--use_gpu=False',
               f'--draw_img_save_dir={PIC_FOLDER}']
    subprocess.run(command,capture_output=True)
    res_txt_path = f'{PIC_FOLDER}{picname}_det_results.txt'
    res_data = None
    with open(res_txt_path,'r') as f:
        res_data = f.read()
    pattern = re.compile(r"\[\[\[.*\]\]\]")
    match = pattern.search(res_data)
    if match:
        extracted_res = match.group()
    else:
        return None,[]
    return 'det_res_'+picname,eval(extracted_res)

