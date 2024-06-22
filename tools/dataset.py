# 这个数据集存在一些错误，我把改正版的存在../dev/finnal_process中
# 存在的问题如下：
# 1、推理慢
# 2、数据集初始化速度慢
# 3、标签是独热编码
# 4、内存占用
# 5、架构较为复杂

from torchvision import transforms
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL
import pathlib
from torch.utils.data import DataLoader, random_split

class HWVocab:
    def __init__(self, data_dir):
        self.labels = []
        self.tlabels = []  # 独热编码后的标签 type: torch.Tensor
        self.char_dict = {}
        self.initialize_dict(data_dir)

    def initialize_dict(self, path):
        data_dir = pathlib.Path(path)
        for i in data_dir.glob('*'):
            if i.is_dir() and i.name not in self.labels:
                self.labels.append(i.name)
        for idx, c in enumerate(self.labels):
            self.char_dict[c] = idx
        self.tlabels = F.one_hot(torch.tensor([self.char_dict[x] for x in self.labels]), num_classes=len(self.labels))
        print(f"Dict initialized successfully, there are {len(self.char_dict)} labels in the dict.")

    def __getitem__(self, c):
        if isinstance(c, list):
            return torch.stack([self.tlabels[self.char_dict[x]] for x in c])
        return self.tlabels[self.char_dict[c]]

    def label2char(self, idx):
        return self.labels[idx]

    def __len__(self):
        return len(self.char_dict)

class HWDataset(Dataset):
    # Hand Writing Dataset
    def __init__(self, data_dir):
        self.transform = transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.ToTensor()
        ])
        self.to_img = transforms.ToPILImage()
        self.data_dir = data_dir
        self.vocab = HWVocab(data_dir)
        self.data_list = self.get_data_list()

    def get_data_list(self):
        data_list = []
        data_dir = pathlib.Path(self.data_dir)
        for file in data_dir.rglob('*'):
            if file.is_file():
                label = self.vocab[file.parent.name]
                data_list.append((file, label))
        print(f"Length of dataset is: {len(data_list)}")
        return data_list

    def get_real_feature(self, file):
        img = PIL.Image.open(file)
        feature = self.transform(img)
        return feature

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.get_real_feature(self.data_list[idx][0]), self.data_list[idx][1]

    def get_img(self, idx):
        item = self.data_list[idx]
        img = self.to_img(self.get_real_feature(item[0]))
        return (self.vocab.label2char(item[1]), img)

def get_data_loader(data_dir, batch_size: int, shuffle: bool, is_train=True, split_ratio=0.5):
    dataset = HWDataset(data_dir=data_dir)
    num_labels = len(dataset.vocab)
    if is_train:
        loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
        return loader, num_labels, dataset
    else:
        ratio = split_ratio
        val_size = int(ratio * len(dataset))
        test_size = len(dataset) - val_size
        val_dataset, test_dataset = random_split(dataset, [val_size, test_size])
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=shuffle)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=shuffle)
        return val_loader, test_loader, num_labels, val_dataset, test_dataset
