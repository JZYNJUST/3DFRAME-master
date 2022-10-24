import torch
import torch.utils.data.dataset as Dataset
import torch.utils.data.dataloader as Dataloader
import numpy as np
import os
# data_path = "./DenoiseData"
# label_path = "./DenoiseLabel"
#创建子类
class DenoiseDataset(Dataset.Dataset):
    #初始化，定义数据内容和标签
    def __init__(self, Data_path, Label_path):
        self.Data_path = Data_path
        self.Label_path = Label_path
        self.input_data = os.listdir(self.Data_path)
        self.label = os.listdir(self.Label_path)
    #返回数据集大小
    def __len__(self):
        return len(self.input_data)
    #得到数据内容和标签
    def __getitem__(self, index):
        input_path = os.path.join(self.Data_path,self.input_data[index])
        label_path = os.path.join(self.Label_path,self.label[index])
        input = torch.Tensor(np.loadtxt(input_path)).view(2,42)
        label = torch.Tensor(np.loadtxt(label_path)).view(2,42)
        return input, label

# dataset = MyDataset(data_path,label_path)
# dataloader = Dataloader.DataLoader(dataset=dataset,batch_size=1,shuffle=False,num_workers=0)
# for i ,item in enumerate(dataloader):
#     data,label = item
#     print(data.size(),label.size())

