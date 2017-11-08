import torch
import numpy as np

class LikelihoodOpt(torch.optim.Optimizer):

    def __init__(self, params, tempreture=0.95, lr=0.5):

        defaults = dict(lr=lr, tempreture=tempreture)
        super(LikelihoodOpt, self).__init__(params, defaults)

    def step(self):

        params = self.param_groups[0]["params"]
        lr = self.param_groups[0]["lr"]
        tempreture = self.param_groups[0]["tempreture"]
        
        