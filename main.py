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

# Define the sigmoid activation function
def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

# Apply the sigmoid activation function to the hidden layer output
a = sigmoid(z)
# print(a.shape)  # Should be (60000, 4)

# Last layer
OUTPUT_LAYER_SIZE = 1

# Initialize weights for the second layer (hidden to output layer) with small random values
weights2 = torch.rand(HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE) * 0.01

# Initialize biases for the output layer with small random values
b2 = torch.rand(OUTPUT_LAYER_SIZE) * 0.01

# Calculate the output of the output layer (before activation)
z2 = a @ weights2 + b2

# Apply the sigmoid activation function to the output layer output
a2 = sigmoid(z2)
# print(a2.shape)  # Should be (60000, 1)

def log_loss(y_true, y_pred):
    epsilon = 1e-15  # Small constant to prevent log(0)
    y_pred = torch.clamp(y_pred, epsilon, 1 - epsilon)  # Clamp predictions to avoid log(0)
    loss = -torch.mean(y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))
    return loss

# Calculate the log loss for the output layer predictions
loss = log_loss(training_data_labels.unsqueeze(1), a2)
print(f'Log Loss: {loss.item()}')