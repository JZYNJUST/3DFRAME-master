import torch
import torch.nn as nn
from tqdm import tqdm
from torch.autograd import Variable

class DenoiseEncoder(nn.Module):
    def __init__(self):
        super(DenoiseEncoder, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(2, 16, 7),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(16, 64, 3),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(64, 256, 3),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(256, 1024, 3),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(1024,512,3,1),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.ConvTranspose1d(512,256,3,1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.ConvTranspose1d(256,128,3,2),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.ConvTranspose1d(128,64,3,2,1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.ConvTranspose1d(64,2,3,2,1,1),
            nn.Sigmoid()
        )
        self.regressor = nn.Sequential(
            nn.Linear(1024,512),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(512,256),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(256,84)
        )
        # self.classify = nn.Sequential(
        #     nn.Linear(1024, 512),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Linear(512, 256),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Linear(256, 128),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Linear(128, 72),
        #     nn.GELU()
        #     # nn.Linear(64, 2),
        # )

    def forward(self, x):
        features = self.features(x)
        # print("features.size = ",features.size())
        features = torch.max(features, 2, keepdim=True)[0]
        # print("features.size = ",features.size())
        # features = features.view(-1, 1024)
        # out = self.regressor(features)
        out = self.decoder(features)
        return out

# net = DenoiseEncoder()
# a = torch.randn(1,2,42)
# b = net(a)
# print(b.size())