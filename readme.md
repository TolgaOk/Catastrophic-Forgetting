# Catastophic Forgeting In Neural Networks

This repository is created to test and implement methods to overcome catastrophic forgetting. The phenomenon of catastrophic forgetting is a very common problem in neural networks. This happens when there are multiple tasks to train or when there are non-homogenous batches. For example, if there are two different tasks, let's say task A and task B. After the network is trained on task A, it will override already learned parameters to learn task B as much as possible. However, the ideal learning algorithm should be able to remember previously learned tasks and should also be able to be trained with non-homogenous batches.

## Testbed

* **MNIST**

    - [x] Non-Homogenuos batches
    - [ ] Permutated tasks

## Dependicies
- Pytorch
