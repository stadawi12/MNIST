# import trained model and test it on the testing dataset
import torch
from preprocess_data import preprocess_data
from neural_network import SimpleNN
from dataloader import MNISTDataset
from torch.utils.data import DataLoader

if __name__ == "__main__":
    # Preprocess the MNIST dataset and obtain the training and testing data
    training_data_images, training_data_labels_one_hot, testing_data_images, testing_data_labels_one_hot = preprocess_data()

    # Create the MNIST dataset using pytorch's Dataset class
    mnist_test_dataset = MNISTDataset(testing_data_images, testing_data_labels_one_hot)

    # Create a data loader for the testing dataset
    test_loader = DataLoader(mnist_test_dataset, batch_size=64, shuffle=False)

    # Create an instance of the SimpleNN class    
    model = SimpleNN([784, 128, 64, 10])  # Define the layer sizes for the neural network

    # Load the trained model parameters from the file
    model.load_state_dict(torch.load("trained_models/mnist_model.pth"))  # Load the model parameters from the file

    # Set the model to evaluation mode
    model.eval()  # Set the model to evaluation mode

    # Test the model on the testing dataset
    with torch.no_grad():  # Disable gradient computation for testing
        test_loss, correct = 0, 0
        for images, labels in test_loader:
            pred = model(images)  # Forward pass through the model to get predictions
            test_loss += torch.nn.CrossEntropyLoss()(pred, labels).item()  # Compute test loss
            correct += (pred.argmax(1) == labels.argmax(1)).type(torch.float).sum().item()  # Count correct predictions

    test_loss /= len(test_loader.dataset)  # Compute average test loss
    accuracy = correct / len(test_loader.dataset) * 100  # Compute accuracy as a percentage

    print(f"Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%")  # Print the test loss and accuracy