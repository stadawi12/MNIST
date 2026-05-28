import torch
from mnist import MNIST

# Load the MNIST dataset
MNIST_dataset = MNIST('dataset')

# Load training and testing data
training_data_images, training_data_labels = MNIST_dataset.load_training()

testing_data_images, testing_data_labels = MNIST_dataset.load_testing()

# Print the shape of the training and testing data
training_example = training_data_images[1]

training_example = torch.tensor(training_example) / 255.0

TRAINING_EXAMPLE_DIMENSION = len(training_example)

HIDDEN_LAYER_SIZE = 4

weights = torch.rand(TRAINING_EXAMPLE_DIMENSION, HIDDEN_LAYER_SIZE) * 0.01

b = torch.rand(HIDDEN_LAYER_SIZE) * 0.01

z = training_example @ weights + b

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

a = sigmoid(z)
print(a)

# Last layer
OUTPUT_LAYER_SIZE = 1

weights2 = torch.rand(HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE) * 0.01
b2 = torch.rand(OUTPUT_LAYER_SIZE) * 0.01

z2 = a @ weights2 + b2
a2 = sigmoid(z2)
print(a2)