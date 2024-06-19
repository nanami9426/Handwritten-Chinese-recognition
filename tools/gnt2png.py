import struct
import os
from PIL import Image
from tqdm import tqdm


def gnt2png(data_path,img_path,label_name):
    """
    data_path:.gnt文件所在目录
    img_path:解析后图片存放目录
    """
    print(f"processing {label_name} data")
    files=os.listdir(data_path) #os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表。
    num=0
    for file in tqdm(files):
        tag = []
        img_bytes = []
        img_wid = []
        img_hei = []
        f=open(data_path+"/"+file,"rb")
        while f.read(4):
            tag_code=f.read(2)
            tag.append(tag_code)
            width=struct.unpack('<h', bytes(f.read(2)))
            height=struct.unpack('<h',bytes(f.read(2)))
            img_hei.append(height[0])
            img_wid.append(width[0])
            data=f.read(width[0]*height[0])
            img_bytes.append(data)
        f.close()
        for k in range(0, len(tag)):
            im = Image.frombytes('L', (img_wid[k], img_hei[k]), img_bytes[k])
            if os.path.exists(img_path + "/" + tag[k].decode('gbk')):
                im.save(img_path + "/" + tag[k].decode('gbk') + "/" + str(num) + ".jpg")
            else:
                os.mkdir(img_path + "/" + tag[k].decode('gbk'))
                im.save(img_path + "/" + tag[k].decode('gbk') + "/" + str(num) + ".jpg")
        num = num + 1
    print(tag.__len__())

    files=os.listdir(img_path)
    n=0
    f=open(f"{label_name}.txt","w") #创建用于训练的标签文件
    for file in files:
        files_d=os.listdir(img_path+"/"+file)
        for file1 in files_d:
            f.write(file+"/"+file1+" "+str(n)+"\n")
        n=n+1

if __name__ == '__main__':
    gnt2png("../sourcedata/HWDB1.1tst_gnt/","../data/test",'tst')
    gnt2png("../sourcedata/HWDB1.1trn_gnt/","../data/train",'trn')