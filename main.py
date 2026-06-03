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
    NUMBER_OF_EPOCHS = 2  # Define the number of epochs for training the model
    HIDDEN_SIZES = [784, 128, 64, 10]  # Define the size of the hidden layer in the neural network

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

    loss_function = CrossEntropyLoss()  # Define the loss function for training the model

    optimiser = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)  # Define the optimizer for updating model parameters

    # Define the training function to train the model on the training dataset
    def train(model, train_loader, loss_function, optimiser):
        
        # Get the total number of samples in the training dataset
        size = len(train_loader.dataset)

        # Set the model to training mode
        model.train()

        # Initialize a variable to accumulate the running loss during training
        running_loss = 0.0

        number_of_examples_passed = 0

        losses = []

        # Iterate through the training data in batches
        for batch, (images, labels) in enumerate(train_loader):
            
            # Forward pass through the model to get predictions
            pred = model(images)

            # Compute the loss between the predictions and the true labels
            loss = loss_function(pred, labels)

            # Append the loss for the current batch to the list of losses for later analysis
            losses.append(loss.item())

            # Accumulate the loss for the current batch to compute the average loss later
            running_loss += loss.item()
            number_of_examples_passed += len(images)

            # Backpropagate the loss and update the model parameters using the optimizer
            loss.backward() 

            # Update the model parameters using the optimizer
            optimiser.step()

            # Clear the gradients for the next iteration
            optimiser.zero_grad()

            if batch == 0:
                print(f"Batch: {batch + 1}, Loss: {loss.item():>8f}")
                print(pred[0].argmax())
                print(labels[0].argmax())

            if batch == 1:
                print(f"Batch: {batch + 1}, Loss: {loss.item():>8f}")
            
            if (batch + 1) % 100 == 0:
                print(f"Batch: {batch + 1}, Loss: {loss.item():>8f}")

        return losses  # Return the list of losses for analysis after training is complete

    # Define the testing function to evaluate the model on the testing dataset
    def test(model, test_loader, loss_function):
        number_of_batches = len(test_loader)
        number_of_samples = len(test_loader.dataset)
        model.eval()  # Set the model to evaluation mode
        correct = 0
        running_test_loss = 0.0
        with torch.no_grad():  # Disable gradient computation for testing
            for images, labels in test_loader:
                pred = model(images)  # Forward pass through the model to get predictions
                test_loss = loss_function(pred, labels)  # Compute the loss for the current batch and accumulate it
                running_test_loss += test_loss.item()  # Accumulate the loss for the current batch to compute the average loss later
                correct += (pred.argmax(1) == labels.argmax(1)).type(torch.float).sum().item()  # Count correct predictions

        running_test_loss /= number_of_batches  # Compute average test loss
        accuracy = correct / number_of_samples  # Compute accuracy of the model on the test dataset
        print(f"Test Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {running_test_loss:>8f} \n")  # Print the test results

    train_losses = []  # Initialize a list to store the losses during training
    
    # Train the model for the specified number of epochs
    for epoch in range(NUMBER_OF_EPOCHS):
        print(f"Epoch {epoch + 1}\n-------------------------------")

        # Train the model on the training dataset and obtain the losses for the current epoch
        train_epoch_losses = train(model, train_loader, loss_function, optimiser)

        # Append the losses for the current epoch to the list of training losses for later analysis
        train_losses.append(train_epoch_losses)

        # Evaluate the model on the test dataset after each epoch to monitor its performance
        test(model, test_loader, loss_function) 

    print("Training complete!")

    train_losses = [loss for epoch_losses in train_losses for loss in epoch_losses]  # Flatten the list of losses from all epochs into a single list
    print(len(train_losses))
    x = [i for i in range(len(train_losses))]  # Create a list of batch numbers corresponding to the training losses
    plt.plot(x, train_losses)  # Plot the training losses over epochs to visualize the training process
    plt.xlabel("Batch Number")  # Label for the x-axis of the plot
    plt.ylabel("Loss")  # Label for the y-axis of the plot
    plt.title("Training Loss")  # Title for the plot
    plt.show()

    # save the trained model to a file
    # torch.save(model.state_dict(), "trained_models/mnist_model.pth")  # Save the model parameters to a file for later use