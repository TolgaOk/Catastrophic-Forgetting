import torch
import numpy as np

class Net(torch.nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        self.fc1 = torch.nn.Linear(28*28, 256)
        self.fc2 = torch.nn.Linear(256, 10)
        self.fc3 = torch.nn.Linear(10, 10)
        self.fc4 = torch.nn.Linear(10, 10)

        self.opt = torch.optim.Adam(self.parameters(), 0.001)

        # self.__init_params()
        self.__init_weights()

    def forward(self, x):

        x = x.view(-1, 28*28)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = (self.fc4(x))
        
        y = torch.nn.functional.sigmoid(x)
        # y = torch.clamp(x, 0.0, 1.0)
        return y, x

    def optimize(self, loss, first_half):

        self.opt.zero_grad()
        loss.backward()
        for name, param in self.named_parameters():
            
            if name.endswith("weight"):
                size = param.size()
                mid_row = 2 if name.startswith("fc4") else size[0]//2
                mid_col = size[1] if name.startswith("fc1") else size[1]//2
                reigon = 1 if first_half else 3
                mid_col = mid_col if first_half else size[1]

                param.grad.data.mul_(mask_param(size, (mid_row, mid_col), reigon).cuda())

            elif name.endswith("bias"):
                size = param.size()[0]
                if first_half:
                    mask_vector = torch.cat([torch.ones(size//2), torch.zeros(size-size//2)]).cuda()
                else:
                    mask_vector = torch.cat([torch.zeros(size//2), torch.ones(size-size//2)]).cuda()
                param.grad.data.mul_(mask_vector)

        self.opt.step()


    def __init_params(self):

        for name, param in self.named_parameters():
            if name.endswith("bias"):
                param.data = torch.zeros(param.size()).cuda()
            else:
                param.data = torch.nn.init.uniform(torch.Tensor(param.size()), a=-0.0003, b=0.0003).cuda()

    def __init_weights(self):
        for name, param in self.named_parameters():
            if not name.startswith("fc1") and name.endswith("weight"):
                size = param.size()
                mid_points = (size[0]//2, size[1]//2) if not name.startswith("fc4") else (2, size[1]//2)
                param.data.mul_(1.0 - mask_param(size, mid_points, 2))

        
def mask_param(size, mid_points, reigon_of_ones):
    """
    1 1 1 - - - -
    1 1 1 - - - -
    1 1 1 - - - -
    - - - - - - - 
    - - - - - - - 
    - - - - - - - 

    Mid points are 3, 3 while the size is (6, 7) and reigon_of_ones is 1
    """
    if reigon_of_ones in (1, 2):
        row_vector = torch.cat([torch.ones(mid_points[0]), torch.zeros(size[0]- mid_points[0])])
    else:
        row_vector = torch.cat([torch.zeros(mid_points[0]), torch.ones(size[0]- mid_points[0])])

    if reigon_of_ones in (1, 3):
        column_vector = torch.cat([torch.ones(mid_points[1]), torch.zeros(size[1]- mid_points[1])])
    else:
        column_vector = torch.cat([torch.zeros(mid_points[1]), torch.ones(size[1]- mid_points[1])])
        
    return row_vector.view(-1, 1)*column_vector.view(1, -1)
