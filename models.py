import torch.nn as nn
import torch

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # First fully connected layer
        self.fc2 = nn.Linear(hidden_size, output_size)  # Second fully connected layer

    # Define the forward pass of the neural network
    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))  # Apply sigmoid activation to the output of the first layer
        x = torch.sigmoid(self.fc2(x))  # Apply sigmoid activation to the output of the second layer
        return x
    
if __name__ == "__main__":
    # Example usage of the SimpleNN class
    input_size = 784  # Number of input features (28x28 pixels for MNIST)
    hidden_size = 20   # Size of the hidden layer
    output_size = 10   # Size of the output layer (for 10-class classification)

    # Create an instance of the SimpleNN class
    model = SimpleNN(input_size, hidden_size, output_size)

    # Print the model architecture
    print(model)