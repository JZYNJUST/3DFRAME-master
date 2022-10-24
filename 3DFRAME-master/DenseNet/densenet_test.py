import os

import cv2.cv2 as cv2
import torch
import torchvision.transforms as transforms

from DenseNet.MyDenseNet import MyDenseNet

model_names = [
    # "../models_die", \
    # "../models_eign1", \
    # "../models_eign2", \
    # "../models_fang", \
    # "../models_rect", \
    # "../models_round", \
    # "../models_tao", \
    "../models_evaldata"
]

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
print("device=", device)

T = transforms.Compose([transforms.ToTensor()])

net = MyDenseNet().eval()
net = net.to(device)

for model_name in model_names:
    print("processing model:", model_name)
    pic_path = model_name+"/pictures"
    net.load_state_dict(torch.load("../models_traindata/net_160.pkl"))

    for i,item in enumerate(os.listdir(pic_path)):
        print("processing... : ",i)
        path = os.path.join(pic_path, item)
        img = cv2.imread(path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        Timg = T(img_gray).to(device)
        Timg = Timg.view(-1, 1, 1024, 1024)
        kp = net(Timg)
        kp = kp.view(42,2)
        for i in range(0, 42):
            x = int(1024 * (kp[i][0]+1)/2)
            y = int(1024 * (kp[i][1]+1)/2)
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.imwrite(model_name+"/DensenetResult/"+item,img)
    torch.cuda.empty_cache()