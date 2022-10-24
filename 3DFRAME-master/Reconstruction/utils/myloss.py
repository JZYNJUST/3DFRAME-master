import cv2
import numpy as np
import torch

def KeyPointLoss(kp2d_real,kp2d_label):
    batch_size = kp2d_label.size()[0]
    point_num = kp2d_label.size()[1]
    loss = torch.zeros(batch_size, point_num).cuda()
    for batch_idx in range(batch_size):
        for point_idx in range(point_num):
            loss[batch_idx][point_idx] = torch.sqrt((kp2d_label[batch_idx][point_idx][0] - kp2d_real[batch_idx][point_idx][0]) ** 2
                                                    + (kp2d_label[batch_idx][point_idx][1] - kp2d_real[batch_idx][point_idx][
                                             1]) ** 2)
    return torch.sum(loss)
    # loss.requires_grad = True
    # return loss

def iou(predict, target, eps=1e-6):
    #remove nan
    #predict[predict!= predict] = 0
    dims = tuple(range(predict.ndimension())[1:])
    intersect = (predict * target).sum(dims)
    union = (predict + target - predict * target).sum(dims) + eps
    return (intersect / union).sum() / intersect.nelement()

def iou_loss_fcn(predict, target):
    return 1 - iou(predict, target)


def shape_loss_fcn(v_ref, v_pred):
    return torch.norm((v_ref - v_pred), 2)

def img_loss_fcn(img_ref, img_pred):
    return torch.norm((img_pred - img_ref), 2)

def sym_loss_fcn(kp3d):
    sym_matrix = np.array([[0,5],[1,4],[28,10],[29,11],[30,12],[31,13],[32,14],[33,15],[34,16],[35,17],[41,23],[40,22],[39,21],[38,20],[37,19],[36,18]])
    loss = 0
    for i in range(16):
        loss+=(kp3d[0][sym_matrix[i][0]][0]+kp3d[0][sym_matrix[i][1]][0])+ \
              (kp3d[0][sym_matrix[i][0]][1]-kp3d[0][sym_matrix[i][1]][1])+ \
              (kp3d[0][sym_matrix[i][0]][2]-kp3d[0][sym_matrix[i][1]][2])
    return loss

def skt_loss_fcn(kp2d):
    kp_idxs = np.array([35,34,33,32,31,30,29,28])
    #计算v1-v8向量
    vectors = torch.zeros((8,2),dtype=torch.float32)
    for i in range(7):
        vectors[i][0] = kp2d[kp_idxs[i+1]][0]-kp2d[kp_idxs[i]][0]
        vectors[i][1] = kp2d[kp_idxs[i+1]][1]-kp2d[kp_idxs[i]][1]
    vectors[7][0] = kp2d[kp_idxs[0]][0] - kp2d[kp_idxs[7]][0]
    vectors[7][1] = kp2d[kp_idxs[0]][1] - kp2d[kp_idxs[7]][1]
    #判断v1是否在第三象限
    r = torch.Tensor([[0,-1],[1,0]])
    while(vectors[0][0]>0 or vectors[0][1]>0):
        for i in range[7]:
            vectors[i] = torch.matmul(r,vectors[i])
    loss = 0
    for i in range(7):
        theta1 = torch.atan2(vectors[i][1],vectors[i][0])
        theta2 = torch.atan2(vectors[i+1][1],vectors[i+1][0])
        delta = theta1-theta2
        if(delta>0):
            loss+=delta
        return loss

# img = cv2.cv2.imread("../yanjing-16_17.jpg",0)
# img = torch.tensor(img,dtype=torch.float32)
# img = img.resize(1,1,1024,1024)
# img = img/255.0
# label = torch.randn(1,1,1024,1024)
# ret = iou_loss(img,img)
# print(ret)