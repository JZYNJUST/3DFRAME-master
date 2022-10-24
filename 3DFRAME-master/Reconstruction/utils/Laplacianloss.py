import torch
import torch.nn as nn
import numpy as np
from neural_renderer.load_obj import load_obj

def createAdjMatrix(faces,vertices):
    point_num = vertices.shape[1]
    adj_matrix = np.zeros((point_num, point_num))
    for i in faces[0]:
        v_0 = i[0]
        v_1 = i[1]
        v_2 = i[2]
        adj_matrix[v_0][v_1] = 1
        adj_matrix[v_0][v_2] = 1
        adj_matrix[v_1][v_0] = 1
        adj_matrix[v_1][v_2] = 1
        adj_matrix[v_2][v_0] = 1
        adj_matrix[v_2][v_1] = 1
    adj_matrix_diag_data = np.sum(adj_matrix, axis=0)

    for col in range(point_num):
        adj_matrix[col][col] = -1 * adj_matrix_diag_data[col]

    adj_matrix = -1 * adj_matrix
    adj_idx = np.nonzero(adj_matrix)
    adj_data = adj_matrix[adj_idx]
    adj_idxt = torch.LongTensor(np.vstack(adj_idx))
    adj_datat = torch.FloatTensor(adj_data)
    adj_sparse = torch.sparse_coo_tensor(adj_idxt, adj_datat, adj_matrix.shape)
    adj_sparse.requires_grad = False
    # adj_sparse = adj_sparse[None,:,:]
    return adj_sparse.cuda()

class Laplacianloss(nn.Module):
    def __init__(self,faces,vertices):
        super(Laplacianloss,self).__init__()
        # print("this is Laploss:",faces.shape)
        self.vertices = vertices
        self.adj_sparse = createAdjMatrix(faces,vertices)

    def forward(self):
        v = self.vertices[0]
        # print(v.shape)
        return torch.norm(torch.matmul(self.adj_sparse,v),2)

# class VerticesOptimization(nn.Module):
#     def __init__(self,vertices,faces):
#         super(VerticesOptimization, self).__init__()
#         self.vertices = nn.Parameter(vertices[None,:,:])
#         self.faces = faces[None,:,:]
#         self.Lap_fcn = Laplacianloss(self.faces,self.vertices)
#     def forward(self):
#         return self.Lap_fcn()

# vertices, faces, vertices_min, vertices_max_1, vertices_max_2 = load_obj("glass_average1.obj")
# print(vertices.shape)
# model = VerticesOptimization(vertices,faces)
# optimizer = torch.optim.Adam(model.parameters())
# for i in range(300):
#     optimizer.zero_grad()
#     loss = model()
#     loss.backward(retain_graph=True)
#     optimizer.step()
#     print(loss,vertices.grad)