import torch
from torch.optim.optimizer import Optimizer

class elastic(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_tensor, psi):
        # ctx is a context object that can be used to stash information
        # for backward computation
        ctx.save_for_backward(psi)
        return input_tensor.clone()

    @staticmethod
    def backward(ctx, grad_output):
        # We return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        # phi = 1.0 / torch.exp(-1.0 * torch.abs(ctx.psi))
        psi, = ctx.saved_variables
        elasticity_function = torch.tanh(torch.abs(psi)*10)
        return elasticity_function.mul(grad_output), torch.mean(grad_output, 0)

class relevance_elastic(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_tensor, psi):
        ctx.save_for_backward(psi)
        return input_tensor.clone()

    @staticmethod
    def backward(ctx, grad_output):
        psi, = ctx.saved_variables
        elasticity_function = torch.exp(-5.0*torch.abs(psi))
        return elasticity_function.mul(grad_output), torch.mean(grad_output, 0)


class elasticity(torch.nn.modules.Module):

    def __init__(self, feature_out, relevance=False):

        super(elasticity, self).__init__()
        self.feature_out = feature_out

        psi_limits = {"a": 0.001, "b":0.2} if relevance else {"a":0.02, "b":3.0}
        psi_init = torch.nn.init.uniform(torch.Tensor(feature_out), **psi_limits)
        self.psi = torch.nn.parameter.Parameter(psi_init)
        
        self.elastic = relevance_elastic if relevance else elastic

    def forward(self, input):
        return self.elastic.apply(input, self.psi)

# class OptimizeElasticity(Optimizer):

#     def __init__(self, params, gamma=0.95, lr=0.5):

#         if not(0.0 <= lr and 1.0 >= lr):
#             lr = 1.0 - gamma

#         defaults = dict(lr=lr, gamma=gamma)
#         super(OptimizeElasticity, self).__init__(params, defaults)

#     def step(self, closure=None, is_abs=False):
        
#         params = self.param_groups[0]["params"]
#         lr = self.param_groups[0]["lr"]
#         gamma = self.param_groups[0]["gamma"]

#         linear_or_abs = torch.abs if is_abs else lambda x: x

#         for param in params:
#             if param.grad is None:
#                 return None 
#                 raise ValueError("Param has a value of None")
#             d_p = param.grad.data
#             size = d_p.size()
#             d_p = linear_or_abs(d_p)
#             d_p = (d_p.view(-1)).view(*size)
#             param.data.mul_(gamma)
#             param.data.add_(d_p.mul_(lr))
        
#         return None

class OptimizeElasticity(Optimizer):

    def __init__(self, params, gamma=0.95, lr=0.5):

        if not(0.0 <= lr and 1.0 >= lr):
            lr = 1.0 - gamma

        defaults = dict(lr=lr, gamma=gamma)
        super(OptimizeElasticity, self).__init__(params, defaults)

    def step(self, closure=None, is_abs=False):
        
        params = self.param_groups[0]["params"]
        lr = self.param_groups[0]["lr"]
        gamma = self.param_groups[0]["gamma"]

        linear_or_abs = torch.abs if is_abs else lambda x: x

        for param in params:
            if param.grad is None:
                return None 
                raise ValueError("Param has a value of None")
            d_p = param.grad.data
            size = d_p.size()
            d_p = linear_or_abs(d_p)
            d_p = (d_p.view(-1)).view(*size)
            param.data.mul_(gamma)
            param.data.add_(d_p.mul_(lr))
        
        return None