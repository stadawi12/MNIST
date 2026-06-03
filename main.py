from preprocess_data import preprocess_data
from dataloader import MNISTDataset
from neural_network import SimpleNN

if __name__ == "__main__":

    from torch.utils.data import DataLoader
    from torch.nn import CrossEntropyLoss
    import matplotlib.pyplot as plt
    import torch

    BATCH_SIZE = 64  # Define the batch size for training and testing
    LEARNING_RATE = 0.1  # Define the learning rate for the optimizer
    NUMBER_OF_EPOCHS = 20  # Define the number of epochs for training the model
    HIDDEN_SIZES = [784, 128, 10]  # Define the size of the hidden layer in the neural network

    # Preprocess the MNIST dataset and obtain the training and testing data
    training_data_images, training_data_labels_one_hot, testing_data_images, testing_data_labels_one_hot = preprocess_data()

    # Create the MNIST dataset using pytorch's Dataset class
    mnist_train_dataset = MNISTDataset(training_data_images, training_data_labels_one_hot) 
    mnist_test_dataset = MNISTDataset(testing_data_images, testing_data_labels_one_hot)

    # Create data loaders for the datasets
    train_loader = DataLoader(mnist_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(mnist_test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Create an instance of the SimpleNN class    
    model = SimpleNN(HIDDEN_SIZES)

    # Define the loss function for training the model
    loss_function = CrossEntropyLoss()  

    # Define the optimizer for updating the model parameters during training
    optimiser = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

    # Define the training function to train the model on the training dataset
    def train(model, train_loader, loss_function, optimiser):
        
        # Get the total number of samples in the training dataset
        size = len(train_loader.dataset)

        # Set the model to training mode
        model.train()

        # Initialize a list to store the losses for each batch during training for later analysis
        losses = []

        # Iterate through the training data in batches
        for batch, (images, labels) in enumerate(train_loader):
            
            # Forward pass through the model to get predictions
            pred = model(images)

            # Compute the loss between the predictions and the true labels
            loss = loss_function(pred, labels)

            # Append the loss for the current batch to the list of losses for later analysis
            losses.append(loss.item())

            # Backpropagate the loss and update the model parameters using the optimizer
            loss.backward() 

            # Update the model parameters using the optimizer
            optimiser.step()

            # Clear the gradients for the next iteration
            optimiser.zero_grad()

            # Print the loss for every 100 batches to monitor the training process            
            if (batch + 1) % 100 == 0:
                print(f"Batch: {batch + 1}, Loss: {loss.item():>8f}")

        return losses  # Return the list of losses for analysis after training is complete

    # Define the testing function to evaluate the model on the testing dataset
    def test(model, test_loader, loss_function):

        # Get the total number of batches in the testing dataset
        number_of_batches = len(test_loader)
        
        # Get the total number of samples in the testing dataset
        number_of_samples = len(test_loader.dataset)

        # Set the model to evaluation mode to disable dropout and batch normalization layers during testing
        model.eval()

        # Initialize a variable to count the number of correct predictions made by the model on the testing dataset
        correct = 0

        # Initialize a variable to accumulate the total loss over the testing dataset for later analysis
        test_loss = 0.0

        # Disable gradient computation during testing to save memory and computational resources since we are not updating the model parameters
        with torch.no_grad():

            # Iterate through the testing data in batches
            for images, labels in test_loader:
                
                # Forward pass through the model to get predictions for the current batch of testing data
                pred = model(images)

                # Compute the loss for the current batch and accumulate it to calculate the average test loss later
                test_loss += loss_function(pred, labels).item()

                # Compare the predicted class with the true labels and count the number of correct predictions for the current batch
                correct += (pred.argmax(1) == labels.argmax(1)).type(torch.float).sum().item()

        # Compute the average test loss by dividing the accumulated test loss by the total number of batches in the testing dataset
        test_loss /= number_of_batches

        # Compute the accuracy of the model on the testing dataset by dividing the number of correct predictions by the total number of samples in the testing dataset
        accuracy = correct / number_of_samples

        print(f"Test Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n")  # Print the test results

        # return the average test loss and accuracy for analysis after testing is complete
        return test_loss, accuracy

    train_losses = []  # Initialize a list to store the losses during training
    test_losses = []  # Initialize a list to store the losses during testing
    
    # Train the model for the specified number of epochs
    for epoch in range(NUMBER_OF_EPOCHS):
        print(f"Epoch {epoch + 1}\n-------------------------------")

        # Train the model on the training dataset and obtain the losses for the current epoch
        train_epoch_losses = train(model, train_loader, loss_function, optimiser)

        # Append the losses for the current epoch to the list of training losses for later analysis
        train_losses.append(train_epoch_losses)

        # Evaluate the model on the test dataset after each epoch to monitor its performance
        test_loss, accuracy = test(model, test_loader, loss_function)

        # Append the test loss for the current epoch to the list of test losses for later analysis
        test_losses.append(test_loss)

    print("Training complete!")

    y = []
    for train_loss in train_losses:
        sum_of_losses = sum(train_loss)  # Calculate the sum of losses for the current epoch
        average_loss = sum_of_losses / len(train_loss)  # Calculate the average loss for
        y.append(average_loss)  # Append the average loss for the current epoch to the list of average losses
    x = [i + 1 for i in range(len(y))]  # Create a list of epoch numbers corresponding to the training losses
    plt.plot(x, y)  # Plot the training losses over epochs to visualize the training process
    plt.plot(x, test_losses)  # Plot the test losses over epochs to visualize the testing process
    plt.xlabel("Epoch")  # Label for the x-axis of the plot
    plt.ylabel("Average Loss")  # Label for the y-axis of the plot
    plt.title("Training and Testing Loss")  # Title for the plot
    plt.legend(["Training Loss", "Testing Loss"])  # Add a legend to the plot to differentiate between training and testing losses
    plt.show()

    # save the trained model to a file
    # torch.save(model.state_dict(), "trained_models/mnist_model.pth")  # Save the model parameters to a file for later use