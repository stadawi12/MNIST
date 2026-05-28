import torch
from mnist import MNIST
import random
import matplotlib.pyplot as plt
import math

MNIST_dataset = MNIST('dataset')

training_data_images, training_data_labels = MNIST_dataset.load_training()

testing_data_images, testing_data_labels = MNIST_dataset.load_testing()

training_example = training_data_images[4]

training_example = torch.tensor(training_example) / 255.0

TRAINING_EXAMPLE_DIMENSION = len(training_example)

HIDDEN_LAYER_SIZE = 4

weights = torch.rand(TRAINING_EXAMPLE_DIMENSION, HIDDEN_LAYER_SIZE) * 0.01

b = torch.rand(HIDDEN_LAYER_SIZE) * 0.01

z = training_example @ weights + b

print(z)

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

a = sigmoid(z)
print(a)