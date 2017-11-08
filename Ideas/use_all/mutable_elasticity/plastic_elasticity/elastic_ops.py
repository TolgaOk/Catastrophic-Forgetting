import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
import numpy as np


class elastic(torch.autograd.Function):
    """ This operation cuts incomming gradients depending on 
    the elasticity of its input. Elasticity values are kept in 
    psi variables. If the elasticity value of a neuron is high 
    it is kept elastic and if the value close to zero then it becomes
    plastic and it resists to any change. However, it has no effect 
    in forward-pass.

    """

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
        elasticity_function = torch.tanh(torch.abs(psi) * 10)
        return elasticity_function.mul(grad_output), torch.mean(grad_output, 0)


class relevance_elastic(torch.autograd.Function):
    """ This operation is very similar to elastic operation, but
    unlike elastic operation, neurons are kept elastic if their 
    elasticity value(psi) is high.
    """

    @staticmethod
    def forward(ctx, input_tensor, psi):
        ctx.save_for_backward(psi)
        return input_tensor.clone()

    @staticmethod
    def backward(ctx, grad_output):
        psi, = ctx.saved_variables
        elasticity_function = torch.exp(-5.0 * torch.abs(psi))
        return elasticity_function.mul(grad_output), torch.mean(grad_output, 0)


class weight_only_elastic(torch.autograd.Function):
    """ This operation only cuts the gradient flowing through
    weights and doesn't affect the gradient flowing through
    layer input. Thus prevents from vanishing gradient. Yet, 
    this operation applies gradient cutting to neurons.
    """

    @staticmethod
    def forward(ctx, input_weight, psi):
        ctx.save_for_backward(psi)
        return input_weight.clone()

    @staticmethod
    def backward(ctx, grad_output):
        psi, = ctx.saved_variables
        elasticity_function = 1.0 - torch.clamp(psi, 0.0, 1.0).view(-1, 1)
        return elasticity_function.mul(grad_output), torch.mean(torch.abs(grad_output), 1)


class bias_only_elastic(torch.autograd.Function):
    """ This is similar to the "weight_only_elastic" operation
    with a little different. This operation applies gradient
    cutting to indivisual weights instead of neurons, since
    bias has the shape of neurons. But this can also be used
    for weights as well.
    """

    @staticmethod
    def forward(ctx, input_bias, psi):
        ctx.save_for_backward(psi)
        return input_bias.clone()

    @staticmethod
    def backward(ctx, grad_output):
        psi, = ctx.saved_variables
        elasticity_function = 1.0 - torch.clamp(psi, 0.0, 1.0)
        return elasticity_function.mul(grad_output), torch.abs(grad_output)


class weight_only_elasticity(torch.nn.modules.Module):
    """ Module of elastic operations those only affect network
    parameters. """

    def __init__(self, parameter_size, bias=False):

        super(weight_only_elasticity, self).__init__()

        psi_init = torch.nn.init.uniform(
            torch.Tensor(*parameter_size), a=0.0, b=0.10)
        self.psi = torch.nn.parameter.Parameter(psi_init)

        self.elastic = weight_only_elastic if not bias else bias_only_elastic

    def forward(self, weight):
        return self.elastic.apply(weight, self.psi)


class elasticity(torch.nn.modules.Module):
    """ Module of elastic operations which is used after
    activation for each layer. """

    def __init__(self, feature_out, relevance=False):

        super(elasticity, self).__init__()
        self.feature_out = feature_out

        psi_limits = {"a": 0.001, "b": 0.2} if relevance else {
            "a": 0.02, "b": 3.0}
        psi_init = torch.nn.init.uniform(
            torch.Tensor(feature_out), **psi_limits)
        self.psi = torch.nn.parameter.Parameter(psi_init)

        self.elastic = relevance_elastic if relevance else elastic

    def forward(self, input):
        return self.elastic.apply(input, self.psi)


class elastic_linear(torch.nn.modules.Module):
    """ Linear module which uses "weight_only_elasticity".
    """

    def __init__(self, feature_in, feature_out):

        super(elastic_linear, self).__init__()
        self.feature_in = feature_in
        self.feature_out = feature_out

        weight_init = torch.nn.init.normal(torch.Tensor(
            feature_out, feature_in), mean=0.0, std=1 / np.sqrt(feature_in))
        self.weight = torch.nn.parameter.Parameter(weight_init)

        bias_init = torch.nn.init.uniform(
            torch.Tensor(feature_out), a=-0.03, b=0.03)
        self.bias = torch.nn.parameter.Parameter(bias_init)

        self.elastic_weight = weight_only_elasticity(
            weight_init.size(), bias=True)
        self.elastic_bias = weight_only_elasticity((feature_out,), bias=True)

    def forward(self, input):

        e_weight = self.elastic_weight(self.weight)
        e_bias = self.elastic_bias(self.bias)

        return torch.nn.functional.linear(input, e_weight, e_bias)


class OptimizeElasticity(Optimizer):
    """ This optimizer is used to update elasticity variables(psi). 
    Elasticty parameters are updated with exponantially decaying function
    which is given below.

        P_(t+1) = P_t*gamma' + |grad|*(1-gamma')
        gamma' = gamma + (1-gamma)*tanh(P_t)

    Thanks to this update, increasing the value of a parameter is faster
    then decreasing it. Therefore, high valued parameters are difficult
    to be decreased. This provides a more permanent plasticity.

    """

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
            gamma_ = torch.tanh(param.data) * (1 - gamma) + gamma

            d_p = param.grad.data
            size = d_p.size()
            param.data.mul_(gamma_)
            param.data.add_(d_p.mul_(1.0 - gamma_))

        return None
