import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from elastic_ops import elastic_linear, OptimizeElasticity


class ElephantNet(nn.Module):
    """ Network with elastic operations. Each layer has an elastic
    operation at its output. This operation has no effect on forward
    pass but it cuts incomming gradients according to the elasticity 
    of each neuron at that layer."""

    def __init__(self):

        super(ElephantNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.elastic1 = elasticity(256, relevance=False)
        self.fc2 = nn.Linear(256, 128)
        self.elastic2 = elasticity(128, relevance=False)
        self.fc3 = nn.Linear(128, 128)
        self.elastic3 = elasticity(128, relevance=False)
        self.fc4 = nn.Linear(128, 10)
        self.elastic4 = elasticity(10, relevance=False)

        self.done_training = False

    def forward(self, x):

        x = x.view(-1, 28 * 28)
        x = self.elastic1(F.relu(self.fc1(x)))
        x = self.elastic2(F.relu(self.fc2(x)))
        x = self.elastic3(F.relu(self.fc3(x)))
        x = self.elastic4(F.log_softmax(self.fc4(x)))
        return x


class ElephantNet2(nn.Module):
    """ Network with elastic linear layers instead of standart
    linear layers. There are two groups of parameters, network 
    weights and elasticity parameters. Network parameters are updated
    with a standart Momentum optimization method, while elasticity 
    parameters are updated with a custom optimizer. Also, elasticity 
    parameters are updated with the gradient of the output of the network 
    instead of loss. This was inspired from the sensitivity works and
    deep taylor decomposition (https://arxiv.org/abs/1512.02479).

        Methods:
            - optimize: Optimizes the network with the input batch
        and corresponding label vector. This method optimizes both
        network weights and elasticity parameters as explained above.
        Also, it returns output as a torch variable and loss value.

            - accuracy: Returns the accuracy of the given output
        variable and target labels.

            - loss: Returns the loss value of given input and target
        tensors.

    """

    def __init__(self):

        super(ElephantNet2, self).__init__()

        self.fc1 = elastic_linear(28 * 28, 256)
        self.fc2 = elastic_linear(256, 128)
        self.fc3 = elastic_linear(128, 128)
        self.fc4 = elastic_linear(128, 10)
        self.done_training = False

        self.neuron_weights = (param for name, param in self.named_parameters()
                               if not name.endswith("psi"))
        self.elasticity_values = (param for name, param in self.named_parameters()
                                  if name.endswith("psi"))

        LR = 0.01
        self.eOptimizer = OptimizeElasticity(
            self.elasticity_values, gamma=0.95, lr=0.05)
        self.optimizer = optim.SGD(
            self.neuron_weights, lr=LR, momentum=0.5, nesterov=True)

    def forward(self, x):

        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # x = F.dropout(x, p=0.80,  training=not self.done_training)
        x = F.log_softmax(self.fc4(x))
        return x

    def optimize(self, input_raw, target_raw):

        input, target = Variable(input_raw.cuda()), Variable(target_raw.cuda())

        self.optimizer.zero_grad()
        output = self(input)
        loss = F.nll_loss(output, target)

        loss.backward()
        self.optimizer.step()

        output = self(input)
        self.eOptimizer.zero_grad()
        label_arange = torch.arange(0, 10)
        one_hot_labels = target_raw.view(-1, 1).float(
        ) == label_arange.expand(target_raw.size()[0], 10)
        auxiliary_loss = torch.exp(-1.0 * loss.view(-1, 1)) * 5.0
        output.backward(one_hot_labels.float().cuda() * (auxiliary_loss.data))
        self.eOptimizer.step(is_abs=True)

        return output, loss

    def accuracy(self, output, target_raw):

        output_guess = np.argmax(output.data.cpu().numpy(), 1)
        numpy_target = target_raw.view(-1).numpy()
        accuracy = (np.mean(output_guess == numpy_target)) * 100.0

        return accuracy

    def loss(self, input_raw, target_raw):

        input, target = Variable(input_raw.cuda()), Variable(target_raw.cuda())

        output = self(input)
        loss = F.nll_loss(output, target)

        return output, loss
