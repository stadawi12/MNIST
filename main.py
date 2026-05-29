import torch
from mnist import MNIST

# Define the maximum pixel value for normalization
MAX_PIXEL_VALUE = 255.0

# Load the MNIST dataset
MNIST_dataset = MNIST('dataset')

# Load training and testing data
training_data_images, training_data_labels = MNIST_dataset.load_training()
testing_data_images, testing_data_labels = MNIST_dataset.load_testing()

# Convert data to torch tensors and normalize the pixel values to the range [0, 1]
training_data_images = torch.tensor(training_data_images, dtype=torch.float32) / MAX_PIXEL_VALUE
training_data_labels = torch.tensor(training_data_labels, dtype=torch.float32) / MAX_PIXEL_VALUE

# print(training_data_images.shape)  # Should be (60000, 784)
# print(training_data_labels.shape)  # Should be (60000,)

# Define the dimensions of the input and hidden layers
TRAINING_EXAMPLE_DIMENSION = len(training_data_images[0])  # This should be 784 for MNIST   

# Define the size of the hidden layer
HIDDEN_LAYER_SIZE = 4

# Initialize weights for the first layer (input to hidden layer) with small random values
weights = torch.rand(TRAINING_EXAMPLE_DIMENSION, HIDDEN_LAYER_SIZE) * 0.01
# print(weights.shape)  # Should be (784, 4)

# Initialize biases for the hidden layer with small random values
b = torch.rand(HIDDEN_LAYER_SIZE) * 0.01

# Calculate the output of the hidden layer (before activation)
z = training_data_images @ weights + b
# print(z.shape)  # Should be (60000, 4)

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

a = sigmoid(z)
# print(a.shape)  # Should be (60000, 4)

# Last layer
OUTPUT_LAYER_SIZE = 1

weights2 = torch.rand(HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE) * 0.01
b2 = torch.rand(OUTPUT_LAYER_SIZE) * 0.01

z2 = a @ weights2 + b2
a2 = sigmoid(z2)
# print(a2.shape)  # Should be (60000, 1)