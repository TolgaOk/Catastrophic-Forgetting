import sys
import torch
import numpy as np
from torch import optim
from network import ElephantNet2
import matplotlib.pyplot as plt

sys.path.append(".")
from Testbed import Mnist

def train():

    ITERATIONS = 400
    TEST_ITERATION = ITERATIONS//5
    BATCH_SIZE = 128
    N_TASKS = 5

    task = Mnist.MnistTasks()
    train_classes, test_classes = task.get_classes()

    model = ElephantNet2()
    model.cuda()

    train_batch = task.heterogen_batches(N_TASKS, train_classes, ITERATIONS, BATCH_SIZE)
    test_batch = task.heterogen_batches(N_TASKS, test_classes, TEST_ITERATION, BATCH_SIZE)
    
    colors = ["b", "r", "g", "c", "m"]

    for i, batch in enumerate(train_batch):
        
        data, target = batch
        output, loss = model.optimize(data, target)
        
        if i % 5 == 4:
            plt.scatter(i, loss.data.cpu().numpy(), s=3, c=colors[i//ITERATIONS])
            plt.pause(0.001)


    for i, batch in enumerate(test_batch):
        
        data, target = batch
        output, loss = model.loss(data, target)
        
        if i % 5 == 4:
            plt.scatter(i + ITERATIONS*N_TASKS, loss.data.cpu().numpy(), s=3, c=colors[i//TEST_ITERATION])
            plt.pause(0.001)



if __name__ == "__main__":
    train()