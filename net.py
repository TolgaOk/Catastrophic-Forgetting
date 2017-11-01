import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn.parameter as Parameter
from elasticity_op import elasticity, OptimizeElasticity
import matplotlib.pyplot as plt
import pickle

BATCH_SIZE = 128
LR = 0.01
ITERATION = 400


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        self.done_training = False

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        # x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, p=0.1,  training=not self.done_training)
        x = self.fc2(x)
        return F.log_softmax(x)

    def save_params(self, name):
        params = self.state_dict()
        pickle.dump(params, open(name, "wb"))
    
    def load_params(self, name):
        state_dict = pickle.load(open(name, "rb"))
        self.load_state_dict(state_dict)


class ElephantNet(nn.Module):
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
        # x = F.dropout(x, p=0.450,  training=not self.done_training)
        x = self.elastic2(F.relu(self.fc2(x)))
        # x = F.dropout(x, p=0.450,  training=not self.done_training)
        x = self.elastic3(F.relu(self.fc3(x)))
        # x = F.dropout(x, p=0.80,  training=not self.done_training)
        x = self.elastic4(F.log_softmax(self.fc4(x)))
        # x = self.fc4(x)
        return x


def data_generator(**kwargs):
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=None, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=BATCH_SIZE, shuffle=False, **kwargs)

    classes = {}

    data, target = next(iter(train_loader))
    indices = torch.from_numpy(np.argsort(target.numpy()))
    length = target.size()[0] // 10

    begin = 0

    for i in range(10):
        end = (torch.sum(target == i)) + begin
        classes[i] = (data[indices[begin:end]],
                      target[indices[begin:end]])
        begin = end

    return classes


def train():
    classes = data_generator()
    model = ElephantNet()

    model.cuda()

    neuron_weights = (param for name, param in model.named_parameters() if not name.startswith("elastic"))
    elasticity_values = (param for name, param in model.named_parameters() if name.startswith("elastic"))

    eOptimizer = OptimizeElasticity(elasticity_values, gamma=0.99, lr=0.01)
    optimizer = optim.SGD(neuron_weights, lr=LR, momentum=0.5)

    label_arange = torch.arange(0, 10)

    index = 0
    accuracy = 0
    colors = ["b", "r", "g", "c", "m"]
    for bonus, addition in enumerate((2, 4, 6, 8, 10, 2, 4, 6, 8, 10)):

        model.done_training = bonus > 4
        for i in range(ITERATION):
            data = torch.from_numpy(np.zeros((BATCH_SIZE, 1, 28, 28)))
            target = torch.from_numpy(np.zeros((BATCH_SIZE), dtype=np.int32))
            for j in range(2):
                j = j + addition - 2
                indices = torch.from_numpy(np.random.randint(
                    0, classes[j][1].size()[0], BATCH_SIZE // 2)).long()
                data[(j%2) * (BATCH_SIZE // 2):((j%2) + 1) * (BATCH_SIZE // 2)] = classes[j][0][indices]
                target[(j%2) * (BATCH_SIZE // 2):((j%2) + 1) * (BATCH_SIZE // 2)] = classes[j][1][indices]

            data_gpu, target_gpu = data.float(), target.long()
            data, target = Variable(data_gpu.cuda()), Variable(target_gpu.cuda())
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)

            if bonus < 5:
                eOptimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
                eOptimizer.step(is_abs=True)
                # output = model(data)
                # one_hot_labels = target_gpu.view(-1, 1).float() == label_arange.expand(target_gpu.size()[0], 10)
                # auxiliary_loss = torch.exp(-loss.view(-1, 1))
                # output.backward(one_hot_labels.float().cuda()*(auxiliary_loss.data))
                # eOptimizer.step(is_abs=True)
            
            else:
                output_guess = np.argmax(output.data.cpu().numpy(), 1)
                numpy_target = target_gpu.view(-1).numpy()
                accuracy += (np.mean(output_guess == numpy_target))*100.0/ITERATION
                if i%ITERATION == ITERATION-1:
                    print(accuracy)
                    accuracy = 0.0

            

            if i % 5 == 4:
                # print("Loss:{} -------------".format(loss.data.cpu().numpy()))
                plt.scatter(i + bonus * ITERATION,
                            loss.data.cpu().numpy(), s=3, c=colors[bonus % 5])
                plt.pause(0.001)


    plt.savefig('temp_elephant_5_drop.png')

def visualization():
    classes = data_generator()
    model = Net()

    model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.5)


    for i in range(0):
        data = torch.zeros(BATCH_SIZE, 1, 28, 28).float()
        target = torch.zeros(BATCH_SIZE).int()
        for j in range(10):
            indices = torch.from_numpy(np.random.randint(   
                0, classes[j][1].size()[0], BATCH_SIZE // 10)).long()
            data[j * (BATCH_SIZE // 10):(j + 1) *( BATCH_SIZE // 10)] = classes[j][0][indices]
            target[(j) * (BATCH_SIZE // 10):(j + 1) *
                    (BATCH_SIZE // 10)] = classes[j][1][indices]

        data, target = data.float().cuda(), target.long().cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)

        loss.backward()
        optimizer.step()
        
        if i % 5 == 4:
            # print("Loss:{} -------------".format(loss.data.cpu().numpy()))
            plt.scatter(i,
                        loss.data.cpu().numpy(), s=3)
            plt.pause(0.0001)
    else:
        plt.close()
        model.save_params("visual_v1.p")
        

    model.load_params("visual_v2.p")

    optimizer.zero_grad()
    data_cpu = classes[0][0][5:7].view(2, 1, 28, 28)
    data = Variable(data_cpu.float().cuda(), requires_grad=True)

    out = model(data)
    label = torch.ones(1).int()*8 == torch.arange(0, 10).view(1, 10).int()
    out.backward(label.float().cuda().expand(2, 10))
    # out.backward(torch.ones(2, 10).cuda())

    heat = (torch.abs(data.grad.data).view(-1, 28*28)).view(-1, 28, 28)
    heat = heat*data.view(-1, 28, 28).data

    plt.subplot(121)
    plt.imshow(heat[1].cpu().view(28, 28).numpy(), cmap="hot")
    plt.subplot(122)
    plt.imshow(data_cpu[1].view(28, 28).numpy(), cmap="gray")
    plt.show()

    print(label)
    print(torch.nn.functional.softmax(out).data.cpu()[1])

def weight_histogram(name="visual_v2"):

    state_dict = pickle.load(open(name, "rb"))
    weights = [np.squeeze(weight.view(-1).cpu().numpy()) for name, weight in state_dict.items() if name.endswith("weight")]

    subplot_number  = 10 + 100*len(weights)
    for w in weights:
        subplot_number += 1
        plt.subplot(subplot_number)
        plt.bar(np.arange(w.shape[0]), np.sort(np.abs(w)))
    plt.show()

if __name__ == "__main__":
    train()