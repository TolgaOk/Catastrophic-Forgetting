import torch
import numpy as np
import sys
from network import Network
from torch.autograd import Variable
import matplotlib.pyplot as plt

sys.path.append(".")
from Testbed import Mnist

def train():

    ITERATIONS = 400
    TEST_ITERATION = ITERATIONS // 5
    BATCH_SIZE = 128
    N_TASKS = 2

    task = Mnist.MnistTasks()
    train_classes, test_classes = task.get_classes()

    model = Network()
    model.cuda()

    train_batch = task.heterogen_batches(
        N_TASKS, train_classes, ITERATIONS, BATCH_SIZE)
    test_batch = task.heterogen_batches(
        N_TASKS, test_classes, TEST_ITERATION, BATCH_SIZE)

    colors = ["b", "r", "g", "c", "m"]

    for i, batch in enumerate(train_batch):

        data, target = batch
        data, target = Variable(data.cuda()), Variable(target.cuda())
        
        output = model(data)
        loss_1 = torch.nn.functional.nll_loss(output, target)
        loss_2 = model.similarity_loss()*torch.exp(-loss_1)

        loss = loss_1*0.05 + loss_2*0.95
        model.optimize(loss*5)

        if i % 5 == 4:
            plt.scatter(i, loss_1.data.cpu().numpy(),
                        s=3, c=colors[i // ITERATIONS])
            plt.pause(0.001)

    for i, batch in enumerate(test_batch):

        data, target = batch
        data, target = Variable(data.cuda()), Variable(target.cuda())
        output = model(data)
        loss_1 = torch.nn.functional.nll_loss(output, target)

        if i % 5 == 4:
            plt.scatter(i + ITERATIONS * N_TASKS, loss_1.data.cpu().numpy(),
                        s=3, c=colors[i // TEST_ITERATION])
            plt.pause(0.001)


if __name__ == "__main__":
    train()