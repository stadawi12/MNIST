import torch.nn as nn
import torch

class SimpleNN(nn.Module):

    def __init__(self, layer_sizes):
        super(SimpleNN, self).__init__()

        self.layers = nn.ModuleList()  # Create a ModuleList to hold the layers of the neural network

        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))

    # Define the forward pass of the neural network
    def forward(self, x):

        for layer in self.layers[:-1]:  # Iterate through all layers except the last one
            x = torch.relu(layer(x))  # Apply ReLU activation function after each layer

        x = torch.softmax(self.layers[-1](x), dim=1)  # Apply softmax activation function to the output layer

        return x
    
if __name__ == "__main__":
    # Example usage of the SimpleNN class
    hidden_sizes = [784, 128, 64, 10]  # Define the layer sizes for the neural network

    # Create an instance of the SimpleNN class
    model = SimpleNN(hidden_sizes)

    # Print the model architecture
    print(model)