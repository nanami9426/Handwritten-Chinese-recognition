import json

import os
import PIL
from flask import Flask,Blueprint,jsonify,request
from torchvision import transforms
import json
import torch
from utils.net import HWNet
from PIL import Image
import numpy as np
# PIC_FOLDER = './dist/uploads/single/'
with open(os.path.join('./dist/model/hw.json'), 'r') as file:
    hw_classes = json.load(file)
NUM_FEATURES = 3755
hwnet = HWNet(NUM_FEATURES)
# hwnet.load_state_dict(torch.load('./dist/model/handwriting.params'))
hwnet.load_state_dict(torch.load('./dist/model/handwriting4.pth'))

hwnet.eval()
to_img = transforms.ToPILImage()


rec_bp = Blueprint('rec_bp',__name__)

def binary_transform(threshold=128):
    def binarize(img):
        img = transforms.functional.to_grayscale(img)
        np_img = np.array(img)
        np_img = (np_img > threshold).astype(np.uint8) * 255
        return Image.fromarray(np_img)
    return transforms.Lambda(binarize)

transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        binary_transform(threshold=128),
        transforms.Resize((48,48)),
        transforms.ToTensor()
    ])


@rec_bp.route('/api/rec_single',methods=['POST'])
def rec():
    # return os.getcwd()
    # print()
    picnames = request.json['picnames']
    words = []
    for picname in picnames:
        picsrc = picname
        img = PIL.Image.open(picsrc)
        feature = transform(img).reshape([1,1,48,48])
        res = hwnet(feature)
        res = hw_classes[res.argmax(1).item()]
        words.append(res)
    return jsonify({"words":words})