import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2 as cv2
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import imageio
from skimage.io import imread, imsave
import neural_renderer as nr
import torchvision.transforms as transforms
from viewpoint import Model_ViewPoint,make_gif
from deform import Deformation_Model
from DenseNet.MyDenseNet import MyDenseNet
from DenoiseEncoder.DenoiseEncoder import DenoiseEncoder
from tqdm import tqdm

def cameraEstimationInit(cameraEstimation,kp):
    cameraEstimation.ref_points = kp
    cameraEstimation.camera_position = nn.Parameter(torch.from_numpy(np.array([1, 10, -14], dtype=np.float32)))
    cameraEstimation.camera_direction = nn.Parameter(torch.from_numpy(np.array([0, 0, 1], dtype=np.float32)))
    renderer = nr.Renderer(camera_mode='look', camera_direction=cameraEstimation.camera_direction).cuda()
    renderer.eye = cameraEstimation.camera_position
    cameraEstimation.renderer = renderer
def deformModelInit(deformModel,cameraDerection,cameraPosition,kp):
    # print(cameraPosition,cameraDerection)
    renderer = nr.Renderer(camera_mode='look', camera_direction=cameraDerection).cuda()
    renderer.eye = cameraPosition
    deformModel.renderer = renderer
    deformModel.kp2d_ref = kp

#define on which device this program run
device = "cuda:0"

#define reconstruction information path
filename_obj = "./average_thick.obj"

#define keypoint detection network refiner network
# detector = MyDenseNet().to(device)
# detector.load_state_dict(torch.load("./net_160.pkl"))
# refiner = DenoiseEncoder().to(device)
# refiner.load_state_dict(torch.load("./denoise_model.pkl"))
cameraEstimation = Model_ViewPoint(filename_obj=filename_obj).to(device)
Deformation = Deformation_Model(filename_obj=filename_obj,filename_ref="./ref_image.jpg").to(device)

#define reference path
img_path = "./pictures"

#define a transform which turn a numpy picture into tensor
T = transforms.Compose([transforms.ToTensor()])

print("Parameters ready!")
for item in os.listdir(img_path):
    # predict keypoint positions
    npImg_path = os.path.join(img_path,item)
    npImg = cv2.imread(npImg_path,1)
    npGray = cv2.cvtColor(npImg,cv2.COLOR_BGR2GRAY)
    _,npRef = cv2.threshold(npGray,10,255,cv2.THRESH_BINARY)
    cv2.imwrite("./ref_image.jpg",cv2.resize(npRef,(256,256)))
    kp = np.loadtxt("./featurepoints/"+item.split('.')[0]+".txt")
    kp = torch.from_numpy(kp)
    kp = kp.view(1,42,2).to(device)
    # cv2.imshow("",npImg)
    # cv2.waitKey(0)
    # break
    # tImg = T(npGray)
    # tImg = tImg.view(1,1,1024,1024).to(device)
    # kp = detector(tImg)
    # kp = (kp + 1) / 2
    # kp = kp.view(-1, 2, 42)
    # kp = kp.detach()
    # kp = refiner(kp)
    # kp = kp.detach().view(1, 42, 2)
    # for i in range(0, 42):
    #     x = int(1024 * kp[0][i][0])
    #     y = int(1024 * kp[0][i][1])
    #     cv2.circle(npImg, (x, y), 5, (0, 0, 255), -1)
    # cv2.imshow("",npImg)
    # cv2.waitKey(0)
    # break
    # estimate camera attitude
    # cameraEstimationInit(cameraEstimation=cameraEstimation,kp=kp)
    # loop = tqdm(range(360))
    # optimizer = torch.optim.Adam(cameraEstimation.parameters(), lr=0.1)
    # for i in loop:
    #     optimizer.zero_grad()
    #     loss = cameraEstimation()
    #     loss.backward()
    #     optimizer.step()
    #     images = cameraEstimation.renderer(cameraEstimation.vertices, cameraEstimation.faces, torch.tanh(cameraEstimation.textures))
    #     image = images.detach().cpu().numpy()[0].transpose(1, 2, 0)
    #     imsave('./tmp/_tmp_%04d.png' % i, image)
    #     loop.set_description('Optimizing (loss %.4f)' % loss.data)
    # make_gif("./viewpoint_gif/"+item.split('.')[0]+".gif")
    # np.savetxt("./viewpoint_gif/direction_"+item.split('.')[0]+".txt",
    #            cameraEstimation.camera_direction.detach().cpu().numpy())
    # np.savetxt("./viewpoint_gif/position_" + item.split('.')[0] + ".txt",
    #            cameraEstimation.camera_position.detach().cpu().numpy())
    D = np.loadtxt("./viewpoint_gif/direction_"+item.split('.')[0]+".txt")
    P = np.loadtxt("./viewpoint_gif/position_" + item.split('.')[0] + ".txt")
    D = torch.tensor(D,dtype=torch.float32).to(device)
    P = torch.tensor(P,dtype=torch.float32).to(device)
    deformModelInit(Deformation,
                    cameraDerection=D,
                    cameraPosition=P,
                    kp=kp)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, Deformation.parameters()),lr=0.01)
    loop = tqdm(range(300))
    for i in loop:
        loop.set_description('Optimizing')
        optimizer.zero_grad()
        loss = Deformation()
        loss.backward()
        optimizer.step()
        images = Deformation.renderer(Deformation.FFD(), Deformation.faces, mode='silhouettes')
        image = images.detach().cpu().numpy()[0]
        imsave('./tmp/_tmp_%04d.png' % i, image)
    make_gif("./deform_gif/"+item.split('.')[0]+".gif")
    nr.save_obj("./"+item.split('.')[0]+".obj", torch.squeeze(Deformation.FFD(), dim=0), Deformation.face2save)
