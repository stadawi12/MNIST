from preprocess_data import preprocess_data
from dataloader import MNISTDataset
from models import SimpleNN

if __name__ == "__main__":

    from torch.utils.data import DataLoader

    # Preprocess the MNIST dataset and obtain the training and testing data
    training_data_images, training_data_labels_one_hot, testing_data_images, testing_data_labels_one_hot = preprocess_data()

    # Create the MNIST dataset using pytorch's Dataset class
    mnist_train_dataset = MNISTDataset(training_data_images, training_data_labels_one_hot) 
    mnist_test_dataset = MNISTDataset(testing_data_images, testing_data_labels_one_hot)

    # Create data loaders for the datasets
    train_loader = DataLoader(mnist_train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(mnist_test_dataset, batch_size=64, shuffle=False)

    # Define the input size, hidden layer size, and output size for the neural network
    input_size = 784  # Number of input features (28x28 pixels for MNIST)
    hidden_size = 20   # Size of the hidden layer
    output_size = 10   # Size of the output layer (for 10-class classification)

    # Create an instance of the SimpleNN class    
    model = SimpleNN(input_size, hidden_size, output_size)

    for batch, (images, labels) in enumerate(train_loader):
        
        pred = model(images)  # Forward pass through the model to get predictions
        print(f"Batch {batch + 1}: Predicted labels shape: {pred.shape}, True labels shape: {labels.shape}")  # Print the shapes of the predicted and true labels
        break  # Break after the first batch for demonstration purposes