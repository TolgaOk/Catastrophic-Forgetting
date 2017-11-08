import torch
import numpy as np

class Network(torch.nn.Module):

    def __init__(self, lr_network=0.01, lr_likelihood=0.01):

        super(Network, self).__init__()

        self.fc1 = torch.nn.Linear(28 * 28, 256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, 128)
        self.fc4 = torch.nn.Linear(128, 10)

        self.network_parameters = self.parameters()
        self.network_parameters_as_vector = torch.cat([param.view(-1) for param in self.parameters()])

        self.likelihood = torch.nn.Linear(self.network_parameters_as_vector.size()[0], 1, bias=False)
        self.likelihood_parameters = (param for name, param in self.named_parameters() if name.startswith("likelihood"))

        self.network_optimizer = torch.optim.SGD(
                                    self.network_parameters, lr=lr_network, momentum=0.5, nesterov=True)
        self.likelihood_optimizer = torch.optim.SGD(
                                    self.likelihood_parameters, lr=lr_likelihood, momentum=0.5, nesterov=True)
        

    def forward(self, x):

        x = x.view(-1, 28*28)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))        
        x = torch.nn.functional.relu(self.fc3(x))
        x = torch.nn.functional.log_softmax(self.fc4(x))
        
        return x

    def optimize(self, loss):
        
        self.network_optimizer.zero_grad()
        self.likelihood_optimizer.zero_grad()
        
        loss.backward()

        self.likelihood_optimizer.step()
        self.network_optimizer.step()
        
    def similarity_loss(self, alpha=0.9):
        
        w = self.network_parameters_as_vector.detach().cuda()
        theta = self.likelihood.weight.view(-1)

        w_len, theta_len = (w.dot(w), theta.dot(theta))

        loss_1 = 1.0 - self.likelihood(w.view(1, -1)) / (torch.sqrt(w_len)*torch.sqrt(theta_len))
        loss_2 = 1.0 - torch.min(torch.cat([w_len, theta_len])) / torch.max(torch.cat([w_len, theta_len]))

        return torch.squeeze(loss_1*alpha + loss_2*(1-alpha))

