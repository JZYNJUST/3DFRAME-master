import numpy as np
import torch
from DenseNet.MyDenseNet import MyDenseNet
from tqdm import tqdm
from dataloader import MyDataset
import torch.utils.data.dataloader as Dataloader
import torchvision.transforms as transforms

model_name = "../models_evaldata"
net = MyDenseNet(num_points=42*2).eval()
net.load_state_dict(torch.load("../models_traindata/net_160.pkl"))
net = net.to("cuda:0")

T = transforms.Compose([transforms.ToTensor()])

pic_path = model_name+"/pictures"
label_path = model_name+"/featurepoints"

dataset = MyDataset(pic_path,label_path)
dataloader = Dataloader.DataLoader(dataset=dataset,batch_size=1,shuffle=False,num_workers=0)

for i,item in tqdm(enumerate(dataloader)):
    print("processing: ",i)
    pic,label = item
    pic = pic.to("cuda:0")
    # print(item.size())
    predict = net(pic)
    predict = predict.detach().to("cpu").numpy()
    label = label.detach().to("cpu").numpy()
    predict = (predict+1)/2
    label = (label+1)/2
    np.savetxt("./DenoiseData/dn_input_"+str(i)+".txt",predict)
    np.savetxt("./DenoiseLabel/dn_label_"+str(i)+".txt",label)
