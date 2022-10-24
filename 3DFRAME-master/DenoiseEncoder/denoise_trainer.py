import torch
import torch.nn as nn
from denoise_dataloader import DenoiseDataset
from DenoiseEncoder import DenoiseEncoder
import torch.utils.data.dataloader as Dataloader
from tqdm import tqdm


data_path = "./DenoiseData"
label_path = "./DenoiseLabel"

dataset = DenoiseDataset(data_path,label_path)
dataloader = Dataloader.DataLoader(dataset=dataset,batch_size=1183,shuffle=True,num_workers=0)

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
print("device=",device)

net = DenoiseEncoder().to(device)
net.load_state_dict(torch.load("./DenoiseModel/denoise_model_800.pkl"))
lr = 0.2
opt = torch.optim.SGD(net.parameters(),lr=1)
lr_dec = 0.998
loss_func = nn.MSELoss()

for epoch in tqdm(range(8000)):
    loss_float = 0
    for index,item in enumerate(dataloader):
        input,label = item
        input = input.to(device)
        label = label.to(device)
        pred = net(input)
        loss = loss_func(pred,label)
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_float += loss.detach().cpu().float().item()
    # lr *= lr_dec
    # print(lr)
    # for param_group in opt.param_groups:
    #     param_group['lr'] = lr
    print("loss = ",loss_float/(index+1))
    if(epoch%100 == 0):
        torch.save(net.state_dict(), "./DenoiseModel/denoise_model_"+str(epoch)+".pkl")
    if loss_float/(index+1) < 1e-4:
        torch.save(net.state_dict(),"./DenoiseModel/denoise_model.pkl")
        break
