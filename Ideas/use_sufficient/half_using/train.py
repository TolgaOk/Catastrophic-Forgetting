import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
from network import Net
from torch.autograd import Variable

sys.path.append(".")
from Testbed import Mnist

def train():

    ITERATIONS = 400
    TEST_ITERATION = ITERATIONS // 2
    BATCH_SIZE = 128
    N_TASKS = 2

    task = Mnist.MnistTasks()
    train_classes, test_classes = task.get_classes()

    model = Net()
    model.cuda()

    experiment_batch = task.heterogen_batches(1, test_classes, 1, 128)
    data, target = next(experiment_batch)
    exp_data, exp_target = Variable(data.cuda()), Variable(target.cuda())

    train_batch = task.heterogen_batches(
        N_TASKS, train_classes, ITERATIONS, BATCH_SIZE)
    test_batch = task.heterogen_batches(
        N_TASKS, test_classes, TEST_ITERATION, BATCH_SIZE)

    colors = ["b", "r", "g", "c", "m"]

    for i, batch in enumerate(train_batch):

        data, target = batch
        data, target = Variable(data.cuda()), Variable(target.cuda())
        output, logit = model(data)

        one_hot = onehot(target)

        loss = torch.mean(torch.pow(output - one_hot, 2)/2)

        model.optimize(loss, (i//ITERATIONS == 0))
        
        if i == ITERATIONS - 1 or i == ITERATIONS*2-1:
            
            output, logit = model(exp_data)
            print(logit[-1], exp_target[-1])


        if i % 5 == 4:
            plt.scatter(i, loss.data.cpu().numpy(),
                        s=3, c=colors[i // ITERATIONS])
            plt.pause(0.001)

    for i, batch in enumerate(test_batch):

        data, target = batch
        data, target = Variable(data.cuda()), Variable(target.cuda())
        output, logit = model(data)
        one_hot = onehot(target)
        loss = torch.mean((output - one_hot)**2/2)

        if i % 5 == 4:
            plt.scatter(i + ITERATIONS * N_TASKS, loss.data.cpu().numpy(),
                        s=3, c=colors[i // TEST_ITERATION])
            plt.pause(0.001)

    plt.savefig("Ideas/use_sufficient/half_using/results/frozen_half.png")

def onehot(variable_target):

    one_hot = torch.arange(0, 10).long().view(1, -1).cuda() == \
        variable_target.data.view(-1, 1)

    return Variable(one_hot.float())

if __name__ == "__main__":
    train()

    