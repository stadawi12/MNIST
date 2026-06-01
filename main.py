from preprocess_data import preprocess_data
from dataloader import MNISTDataset
from models import SimpleNN

if __name__ == "__main__":

    from torch.utils.data import DataLoader
    from torch.nn import CrossEntropyLoss
    import torch

    BATCH_SIZE = 32  # Define the batch size for training and testing
    LEARNING_RATE = 0.1  # Define the learning rate for the optimizer
    NUMBER_OF_EPOCHS = 15  # Define the number of epochs for training the model

    # Preprocess the MNIST dataset and obtain the training and testing data
    training_data_images, training_data_labels_one_hot, testing_data_images, testing_data_labels_one_hot = preprocess_data()

    # Create the MNIST dataset using pytorch's Dataset class
    mnist_train_dataset = MNISTDataset(training_data_images, training_data_labels_one_hot) 
    mnist_test_dataset = MNISTDataset(testing_data_images, testing_data_labels_one_hot)

    # Create data loaders for the datasets
    train_loader = DataLoader(mnist_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(mnist_test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Define the input size, hidden layer size, and output size for the neural network
    input_size = 784  # Number of input features (28x28 pixels for MNIST)
    hidden_size = 1000   # Size of the hidden layer
    output_size = 10   # Size of the output layer (for 10-class classification)

    # Create an instance of the SimpleNN class    
    model = SimpleNN(input_size, hidden_size, output_size)

    loss_function = CrossEntropyLoss()  # Define the loss function for training the model

    optimiser = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)  # Define the optimizer for updating model parameters

    def train(model, train_loader, loss_function, optimiser):
        size = len(train_loader.dataset)  # Get the total number of samples in the training dataset
        model.train()  # Set the model to training mode
        for batch, (images, labels) in enumerate(train_loader):
            
            pred = model(images)  # Forward pass through the model to get predictions
            loss = loss_function(pred, labels)  # Compute the loss between predictions and true labels
            
            loss.backward()  # Backpropagate the loss to compute gradients
            optimiser.step()  # Update model parameters using the optimizer
            optimiser.zero_grad()  # Clear the gradients for the next iteration
            if batch % 100 == 0:
                print(f"Batch: {batch + 1}, Batch Loss: {loss.item()}")  # Print the loss every 100 batches

    def test(model, test_loader, loss_function):
        size = len(test_loader.dataset)  # Get the total number of samples in the testing dataset
        model.eval()  # Set the model to evaluation mode
        test_loss, correct = 0, 0
        with torch.no_grad():  # Disable gradient computation for testing
            for images, labels in test_loader:
                pred = model(images)  # Forward pass through the model to get predictions
                test_loss += loss_function(pred, labels).item()  # Accumulate the test loss
                correct += (pred.argmax(1) == labels.argmax(1)).type(torch.float).sum().item()  # Count correct predictions

        test_loss /= size  # Compute average test loss
        accuracy = correct / size  # Compute accuracy of the model on the test dataset
        print(f"Test Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n")  # Print the test results

    for epoch in range(NUMBER_OF_EPOCHS):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        train(model, train_loader, loss_function, optimiser)  # Train the model using the defined training function
        test(model, test_loader, loss_function)  # Test the model using the defined testing function
    print("Training complete!")