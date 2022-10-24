import os

import cv2.cv2 as cv2
import torch
import torchvision.transforms as transforms

from DenoiseEncoder import DenoiseEncoder
from DenseNet.MyDenseNet import MyDenseNet

model_name = "../models_tao"

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
print("device=", device)

T = transforms.Compose([transforms.ToTensor()])
pic_path = model_name+"/pictures"
net = MyDenseNet().eval()
net.load_state_dict(torch.load(model_name + "/net_150.pkl"))
net = net.to(device)
denoiser = DenoiseEncoder().eval()
denoiser.load_state_dict(torch.load(model_name+"/denoise_model.pkl"))
denoiser = denoiser.to(device)

for i,item in enumerate(os.listdir(pic_path)):
    print("processing... : ", i)
    path = os.path.join(pic_path, item)
    img = cv2.imread(path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    Timg = T(img_gray).to(device)
    Timg = Timg.view(-1, 1, 1024, 1024)
    kpn = net(Timg)
    kpn = kpn.view(-1, 2, 42)
    kp = denoiser(kpn)
    kp = kp.view(42, 2)
    for i in range(0, 42):
        x = int(1024 * kp[i][0])
        y = int(1024 * kp[i][1])
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
    cv2.imwrite(model_name + "/denoise_result/" + item, img)
