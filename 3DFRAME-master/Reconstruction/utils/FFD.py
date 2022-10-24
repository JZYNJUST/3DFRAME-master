import torch.nn as nn
import torch
import numpy as np

def Bernstein(i,l,s):
    num = np.math.factorial(l)*(s**i)*((1-s)**(l-i))
    den = np.math.factorial(i)*np.math.factorial(l-i)
    return num/den

def createControlPoints(kx,ky,kz):
    lx = np.linspace(start=0, stop=1, num=kx, endpoint=True)
    ly = np.linspace(start=0, stop=1, num=ky, endpoint=True)
    lz = np.linspace(start=0, stop=1, num=kz, endpoint=True)
    p = torch.zeros((3, kx, ky, kz), dtype=torch.float32)
    for i1 in range(kx):
        for i2 in range(ky):
            for i3 in range(kz):
                # print(i1,i2,i3)
                p[0][i1][i2][i3] = lx[i1]
                p[1][i1][i2][i3] = ly[i2]
                p[2][i1][i2][i3] = lz[i3]
    return p

def normalize_zero_to_one(vertices):
    # input vertices shape b*n*c
    vertices = (vertices+1)/2
    return vertices

def normalize_minus_one_to_one(vertices):
    # input vertices shape b*n*c
    vertices = 2*vertices-1
    return vertices

class FFDStructureLoss(nn.Module):
    def __init__(self,p):
        super(FFDStructureLoss,self).__init__()
        # control point p's shape is 3*k*k*k ,box controller as default
        self.px = p[0, :, :, :]
        self.py = p[1, :, :, :]
        self.pz = p[2, :, :, :]
        _,self.kx,self.ky,self.kz = p.shape

    def forward(self):
        d_px = torch.zeros(self.px.shape,dtype=torch.float32)
        d_py = torch.zeros(self.py.shape,dtype=torch.float32)
        d_pz = torch.zeros(self.pz.shape,dtype=torch.float32)
        zero = torch.tensor([0],dtype=torch.float32)
        for i in range(self.kx - 1):
            d_px[i, :, :] = self.px[i, :, :] - self.px[i + 1, :, :]
        for i in range(self.ky - 1):
            d_py[:, i, :] = self.py[:, i, :] - self.py[:, i + 1, :]
        for i in range(self.kz - 1):
            d_pz[:, :, i] = self.pz[:, :, i] - self.pz[:, :, i + 1]
        loss = torch.sum(torch.where(d_px > zero, d_px, zero)) + \
               torch.sum(torch.where(d_py > zero, d_py, zero)) + \
               torch.sum(torch.where(d_pz > zero, d_pz, zero))
        return loss

class FFD(nn.Module):
    def __init__(self,vertices,kx,ky,kz):
        super(FFD, self).__init__()
        # input vertices b*n*c
        # input devide number k is an int
        self.p = nn.Parameter(createControlPoints(kx,ky,kz).cuda())  # p is control points
        self.p.requires_grad = True
        self.tar = normalize_zero_to_one(vertices[None,:,:]).permute(0, 2, 1).cuda()  # tar is controlled vertices
        self.kx = kx  # k is divide number,total number of control points is kx*ky*kz
        self.ky = ky
        self.kz = kz

    def forward(self):
        # print(self.p)
        q = torch.zeros(self.tar.shape,dtype=torch.float32).cuda()  # q is the calculated new vertices
        for i1 in range(self.kx):
            for i2 in range(self.ky):
                for i3 in range(self.kz):
                    # print("p[:,i1,i2,i3].shape = ",p[:,i1,i2,i3].shape)
                    q += Bernstein(i1, self.kx - 1, self.tar[:,0:1,:]) * \
                         Bernstein(i2, self.ky - 1, self.tar[:,1:2,:]) * \
                         Bernstein(i3, self.kz - 1, self.tar[:,2:,:]) * \
                         self.p[:,i1,i2,i3].view(3, 1)
        # make q distribute from 0 to 1,according with neural renderer's input
        q = normalize_minus_one_to_one(q)
        q = q.permute(0,2,1)
        return q
