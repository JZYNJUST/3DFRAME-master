import torch
import torch.nn as nn
import numpy as np
from neural_renderer.load_obj import load_obj

class SymmetricLoss(nn.Module):
    def __init__(self, mapping_table_path="C:/PythonProject/task/Reconstruction/utils/mapping_table.npy"):
        super(SymmetricLoss, self).__init__()
        self.mapping_table = np.load(mapping_table_path)
        # self.mapping_table.requires_grad=False
    def forward(self,vt):
        vt_flatten = vt.view(-1,3)
        mapping_size = self.mapping_table.shape[0]
        loss = 0
        for mapping in self.mapping_table:
            map_from = mapping[0]
            map_to = mapping[1]
            v_to=vt_flatten[map_to]
            x_to = v_to[0]
            y_to = v_to[1]
            z_to = v_to[2]
            v_from=vt_flatten[map_from]
            x_from = v_from[0]
            y_from = v_from[1]
            z_from = v_from[2]
            distance = torch.sqrt((x_to+x_from)**2 + (y_to-y_from)**2 + (z_to-z_from)**2)
            loss += distance
        return loss/mapping_size