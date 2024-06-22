from net import HWNet
import torch
from torchvision import transforms
import PIL
import json
with open('hw.json', 'r') as file:
    hw_classes = json.load(file)
NUM_FEATURES = 3755
hwnet = torch.load("../model/handwriting_rec.pth")
to_img = transforms.ToPILImage()
transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48,48)),
        transforms.ToTensor()
    ])
def run():
    img = PIL.Image.open('../data/test/鸡/12.jpg')
    feature = transform(img).reshape([1,1,48,48])
    res = hwnet(feature)
    res_t3 = [x.item() for x in list(torch.topk(res,3).indices[0])]
    res_char = [hw_classes[x] for x in res_t3]
    return res_char

if __name__ == '__main__':
    res = run()
    for i in res:
        print(i)