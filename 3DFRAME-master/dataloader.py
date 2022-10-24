import torch
import torch.utils.data.dataset as Dataset
import torchvision.transforms as transforms
import numpy as np
import os
import cv2 as cv

T = transforms.Compose([transforms.ToTensor()])

#创建子类
class MyDataset(Dataset.Dataset):
    #初始化，定义数据内容和标签
    def __init__(self, Data_path, Label_path):
        self.Data_path = Data_path
        self.Label_path = Label_path
        self.dataset = os.listdir(self.Data_path)
        self.label = os.listdir(self.Label_path)
    #返回数据集大小
    def __len__(self):
        return len(self.dataset)
    #得到数据内容和标签
    def __getitem__(self, index):
        img_path = os.path.join(self.Data_path,self.dataset[index])
        label_path = os.path.join(self.Label_path,self.label[index])
        img = cv.imread(img_path,0)
        img = T(img)
        label = torch.Tensor(np.loadtxt(label_path)).view(-1)
        label = label * 2 - 1
        return img, label

class DistMapDataset(Dataset.Dataset):
    # 初始化，定义数据内容和标签
    def __init__(self, path):
        self.path = path
        self.dataset = os.listdir(self.path)

    # 返回数据集大小
    def __len__(self):
        return len(self.dataset)

    # 得到数据内容和标签
    def __getitem__(self, index):
        distmap_path = os.path.join(self.path,self.dataset[index])
        distmap = np.load(distmap_path)
        distmap = torch.tensor(distmap,dtype=torch.float32)
        return distmap
# dataset = MyDataset(data_path,label_path)
# dataloader = Dataloader.DataLoader(dataset=dataset,batch_size=2,shuffle=False,num_workers=0)
# for i ,item in enumerate(dataloader):
#     data,label = item
#     print(data.size(),label.size())

