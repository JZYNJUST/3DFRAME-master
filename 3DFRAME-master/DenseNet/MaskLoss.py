import torch
import torch.nn as nn

class MaskLoss(nn.Module):
    def __init__(self):
        super(MaskLoss, self).__init__()
        self.width = 1024
        self.height = 1024
    def forward(self,predict,distmap):
        batch_size = distmap.size()
        batch_size =batch_size[0]
        loss_array = torch.zeros((batch_size,42),dtype=torch.float32)
        predict = predict.view(batch_size,42, 2)
        loss = 0
        for batch_idx in range(batch_size):
            for idx, point in enumerate(predict[batch_idx]):
                x = self.width * (point[0]+1)/2
                y = self.height * (point[1]+1)/2
                x_int = int(x)
                y_int = int(y)
                x_one = torch.div(point[0],point[0],rounding_mode='trunc')
                y_one = torch.div(point[1],point[1],rounding_mode='trunc')
                # loss_array[batch_idx][idx]=distmap[batch_idx][x_int][y_int]*x_one*y_one
                loss += distmap[batch_idx][x_int][y_int]*x_one*y_one
        loss = loss/batch_size/42
        # loss = torch.sum(loss_array)/42/batch_size
        # loss.requires_grad=True
        return loss