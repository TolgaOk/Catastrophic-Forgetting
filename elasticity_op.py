import torch
from torch.optim.optimizer import Optimizer
import numpy as np

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

class weight_only_elastic(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_weight, psi):
        ctx.save_for_backward(psi)
        return input_weight.clone()

    @staticmethod
    def backward(ctx, grad_output):
        psi, = ctx.saved_variables
        elasticity_function = 1.0 -torch.clamp(psi, 0.0, 1.0).view(-1, 1)
        return elasticity_function.mul(grad_output), torch.mean(torch.abs(grad_output), 1)

class bias_only_elastic(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_bias, psi):
        ctx.save_for_backward(psi)
        return input_bias.clone()

    @staticmethod
    def backward(ctx, grad_output):
        psi, = ctx.saved_variables
        elasticity_function = 1.0 -torch.clamp(psi, 0.0, 1.0)
        return elasticity_function.mul(grad_output), torch.abs(grad_output)

class weight_only_elasticity(torch.nn.modules.Module):

    def __init__(self, feature_out, bias=False):

        super(weight_only_elasticity, self).__init__()
        self.feature_out = feature_out

        psi_init = torch.nn.init.uniform(torch.Tensor(feature_out), a=0.0, b=0.10)
        self.psi = torch.nn.parameter.Parameter(psi_init)

        self.elastic = weight_only_elastic if not bias else bias_only_elastic

    def forward(self, weight):
        return self.elastic.apply(weight, self.psi)

class elastic_linear(torch.nn.modules.Module):

    def __init__(self, feature_in, feature_out):

        super(elastic_linear, self).__init__()
        self.feature_in = feature_in
        self.feature_out = feature_out
        
        weight_init = torch.nn.init.normal(torch.Tensor(feature_out, feature_in), mean=0.0, std=1/np.sqrt(feature_in))
        self.weight = torch.nn.parameter.Parameter(weight_init)

        bias_init = torch.nn.init.uniform(torch.Tensor(feature_out), a=-0.03, b=0.03)
        self.bias = torch.nn.parameter.Parameter(bias_init)

        self.elastic_weight = weight_only_elasticity(feature_out)
        self.elastic_bias = weight_only_elasticity(feature_out, bias=True)

    def forward(self, input):

        e_weight = self.elastic_weight(self.weight)
        e_bias = self.elastic_bias(self.bias)
        
        return torch.nn.functional.linear(input, e_weight, e_bias)

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
                raise ValueError("Param has a value of None")
            gamma_ = torch.tanh(param.data)*(1 - gamma) + gamma
            
            d_p = param.grad.data
            size = d_p.size()
            param.data.mul_(gamma_)
            param.data.add_(d_p.mul_(1.0 - gamma_))

        return None