import re
import subprocess
import cv2
import numpy as np
PIC_FOLDER = '../dist/uploads/'
import cv2
import uuid
import os

def get_warp(picname,detection_boxes):
    picsrc = PIC_FOLDER + picname
    image = cv2.imread(picsrc)
    if image is None:
        raise ValueError("Image not loaded. Please check the image path.")
    warps = []
    for i, box in enumerate(detection_boxes):
        # 计算最小外接矩形
        box = np.array(box, dtype=np.float32)
        rect = cv2.minAreaRect(box)
        box_points = cv2.boxPoints(rect)
        box_points = np.int0(box_points)

        # 获取最小外接矩形的宽度和高度
        width = int(rect[1][0])
        height = int(rect[1][1])

        # 计算透视变换矩阵
        src_pts = box_points.astype("float32")
        dst_pts = np.array([[0, height-1], [0, 0], [width-1, 0], [width-1, height-1]], dtype="float32")
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        # 进行透视变换并裁剪图像
        warped = cv2.warpPerspective(image, M, (width, height))
        warped_name = f'det_{picname}_{i+1}.jpg'
        warps.append(warped_name)

        respicsrc = PIC_FOLDER + '/warps/' + warped_name
        cv2.imwrite(respicsrc, warped)
        # ####################
        warped = cv2.imread(respicsrc)
        height, width = warped.shape[:2]
        if width < height:
            warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
        command = ['python','tools/infer/predict_cls.py',
                    f'--image_dir={respicsrc}',
                    f'--cls_model_dir=./ch_ppocr_mobile_v2.0_cls_infer',
                    f'--use_gpu=False']
        res = subprocess.run(command,capture_output=True)
        stdout_str = res.stdout.decode()
        match = re.search(r"Predicts of .*?\[(.*?)\]", stdout_str)
        if match:
            direction_score = match.group(1)
            direction, score = eval(direction_score)
            print(f"Direction: {direction}, Score: {score}")
            if direction == '0' and int(score)>0.7:
                print("000000000000000000000000000000",respicsrc)
                rotated = warped
            elif direction == '180' and int(score)>0.7:
                print("888888888888888888888888888888",respicsrc)
                rotated = cv2.rotate(warped, cv2.ROTATE_180)
            else: 
                rotated = warped
            cv2.imwrite(respicsrc, rotated)
        else:
            print("没有找到方向和得分")
    return warps



def segment_characters(image_path, output_folder):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))  # 调整核大小
    morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))  # 细化字符连接
    morphed = cv2.dilate(morphed, kernel, iterations=1)
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
    os.makedirs(output_folder, exist_ok=True)
    line = []
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        char_image = image[y:y+h, x:x+w]
        unique_id = str(uuid.uuid4())
        char_image_path = os.path.join(output_folder, f'{unique_id}_char_{i}.png')
        cv2.imwrite(char_image_path, char_image)
        print(f'Saved character {i} to {char_image_path}')
        line.append(char_image_path)
    return line


def get_single(warps):
    lines = []
    for warp in warps:
        c1 = PIC_FOLDER+'warps/'+warp
        c2 = PIC_FOLDER + 'single/'
        line = segment_characters(c1,c2)
        lines.append(line)
        # lines.append((c1,c2))
    return lines