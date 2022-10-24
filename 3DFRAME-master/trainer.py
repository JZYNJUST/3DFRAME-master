import torch
import torch.nn as nn
from dataloader import MyDataset,DistMapDataset
from DenseNet.MyDenseNet import MyDenseNet
from DenseNet.MaskLoss import MaskLoss
import torch.utils.data.dataloader as Dataloader
from tqdm import tqdm
model_name = "round"
pic_path = "./models/models_"+model_name+"/pictures"
label_path = "./models/models_"+model_name+"/featurepoints"
distmap_path = "./models/models_"+model_name+"/distmap"

dataset = MyDataset(pic_path,label_path)
distmapset = DistMapDataset(distmap_path)
dataloader = Dataloader.DataLoader(dataset=dataset,batch_size=1,shuffle=False,num_workers=0)
distloader = Dataloader.DataLoader(dataset=distmapset,batch_size=1,shuffle=False,num_workers=0)

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
print("device=",device)

net = MyDenseNet().cuda()
lr = 0.05
lr_dec = 0.92
opt = torch.optim.SGD(net.parameters(),lr=lr)
mse_func = nn.MSELoss()
dist_func = MaskLoss()

print("training.........................")
for epoch in tqdm(range(160)):
    for item in zip(dataloader,distloader):
        item1,item2 = item
        data,label = item1
        distmap = item2
        data = data.to(device)
        label = label.to(device)
        pred = net(data)
        loss = mse_func(pred,label)
        # loss = dist_func(pred, distmap)
        print("loss=",loss)
        opt.zero_grad()
        loss.backward()
        opt.step()
    lr *= lr_dec
    for param_group in opt.param_groups:
        param_group['lr'] = lr

torch.save(net.state_dict(), "./models_"+model_name+"/net_" + str(160) + ".pkl")