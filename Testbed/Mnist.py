import numpy as np
import torch
from torchvision import datasets, transforms

class MnistTasks(object):

    def __init__(self):
        self.train_loader, self.test_loader = self.load_data()

    def load_data(self, **kwargs):

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
            batch_size=None, shuffle=False, **kwargs)
        return train_loader, test_loader

    def get_classes(self, train_loader=None, test_loader=None):

        if test_loader == None:
            try:
                train_loader = self.train_loader
                test_loader = self.test_loader
            except AttributeError:
                raise ValueError("Argument: <train_loader> or <test_loader> must be defined.")

        train_and_test_classes = []
        for loader in (train_loader, test_loader):
            classes = {}

            data, target = next(iter(loader))
            indices = torch.from_numpy(np.argsort(target.numpy()))
            length = target.size()[0] // 10

            begin = 0

            for i in range(10):
                end = (torch.sum(target == i)) + begin
                classes[i] = (data[indices[begin:end]],
                                target[indices[begin:end]])
                begin = end

            train_and_test_classes.append(classes)

        return train_and_test_classes

    def heterogen_batches(self, n_tasks, classes, iterations=400, batch_size=128):
        
        assert n_tasks > 1 or n_tasks < 6, "Number of tasks must be in the range of (2, 5)"
        labels = (0, 2, 4, 6, 8)
        
        for bonus, addition in enumerate(labels[:n_tasks]):

            for i in range(iterations):
                data = torch.zeros(batch_size, 1, 28, 28)
                target = torch.LongTensor(batch_size).zero_()
                for j in range(2):
                    j = j + addition
                    indices = torch.from_numpy(np.random.randint(
                        0, classes[j][1].size()[0], batch_size // 2)).long()
                    data[(j%2) * (batch_size // 2):((j%2) + 1) * (batch_size // 2)] = classes[j][0][indices]
                    target[(j%2) * (batch_size // 2):((j%2) + 1) * (batch_size // 2)] = classes[j][1][indices]

                yield data, target

