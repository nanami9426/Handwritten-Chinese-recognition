import torch
from torchvision import transforms
import PIL
import json
import time
from net import HWNet
from PIL import Image
import numpy as np
def binary_transform(threshold=128):
    def binarize(img):
        img = transforms.functional.to_grayscale(img)
        np_img = np.array(img)
        np_img = (np_img > threshold).astype(np.uint8) * 255
        return Image.fromarray(np_img)
    return transforms.Lambda(binarize)

with open('hw.json', 'r') as file:
    hw_classes = json.load(file)
NUM_FEATURES = 3755
hwnet = HWNet(NUM_FEATURES)
# hwnet = torch.load("./handwriting_rec.pth")
hwnet.load_state_dict(torch.load("./params.pth"))
hwnet.eval()
to_img = transforms.ToPILImage()
transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        binary_transform(threshold=128),
        transforms.Resize((48,48)),
        transforms.ToTensor()
    ])
def run():
    img = PIL.Image.open('sampledata/3.jpg')
    feature = transform(img).reshape([1,1,48,48])

    start = time.time()
    res = hwnet(feature)
    resimg = to_img(feature.reshape([1,48,48]))
    # resimg.show()
    end = time.time()
    print(f"Inference time consumption : {(end-start):.4f}s")
    res_t3 = [x.item() for x in list(torch.topk(res,3).indices[0])]
    res_char = [hw_classes[x] for x in res_t3]
    return res_char

if __name__ == '__main__':
    res = run()
    for i in res:
        print(i)