from torchvision import transforms
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL
import pathlib
from torch.utils.data import DataLoader,random_split


class HWVocab:
    def __init__(self,data_dir):
        self.lables = []
        self.tlables = [] # 独热编码后的标签 type: torch.Tensor
        self.char_dict = {}
        self.initialize_dict(data_dir)
    def initialize_dict(self,path):
        data_dir = pathlib.Path(path)
        for i in data_dir.glob('*'):
            if i not in self.lables:
                self.lables.append(i.name)
        for idx,c in enumerate(self.lables):
            self.char_dict[c] = idx
        self.tlables = F.one_hot(torch.tensor([self.char_dict[x] for x in self.lables]))
        print(f"dict initialized successfully,there's {len(self.char_dict)} lables in the dict.")
        
    def __getitem__(self,c):
        if isinstance(c,list):
            return [self.tlables[self.char_dict[x]] for x in c]
        return self.tlables[self.char_dict[c]]
        
    def lable2char(self,idx):
        if isinstance(idx,torch.Tensor):
            return self.lables[idx.argmax().item()]
        return self.lables[idx]
    
    def __len__(self):
        return len(self.char_dict)

class HWDataset(Dataset):
    # Hand Writting Dataset
    def __init__(self,data_dir):
        self.transform = transforms.Compose([
            transforms.Resize((48,48)),
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
                lable = self.vocab[file.parent.name]
                # img = PIL.Image.open(file)
                # feature = self.transform(img)
                data_list.append((file,lable))
        print(f"lenth of dataset is : {len(data_list)}")
        return data_list
    
    def get_real_feature(self,file):
        img = PIL.Image.open(file)
        feature = self.transform(img)
        return feature
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self,idx):
        return self.get_real_feature(self.data_list[idx][0]),self.data_list[idx][1]
    
    def get_img(self,idx):
        item = self.data_list[idx]
        img = self.to_img(self.get_real_feature(item[0]))
        return (self.vocab.lable2char(item[1]),img)


def get_data_loader(data_dir,batch_size:int,shuffle:bool,is_train=True,split_ratio=0.5):
    dataset = HWDataset(data_dir=data_dir)
    num_labels = len(dataset.vocab)
    if is_train:
        loader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=shuffle)
        return loader,num_labels,dataset
    else:
        ratio = split_ratio
        val_size = int(ratio*len(dataset))
        tes_size = len(dataset) - val_size
        val_dataset,test_dataset = random_split(dataset,[val_size,tes_size])
        val_loader = DataLoader(dataset=val_dataset,batch_size=batch_size,shuffle=shuffle)
        test_loader = DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=shuffle)
        return val_loader,test_loader,num_labels,val_dataset,test_dataset