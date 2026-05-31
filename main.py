from preprocess_data import preprocess_data
from dataloader import MNISTDataset

if __name__ == "__main__":
    # Preprocess the MNIST dataset and obtain the training and testing data
    training_data_images, training_data_labels_one_hot, testing_data_images, testing_data_labels_one_hot = preprocess_data()

    # Create the MNIST dataset using pytorch's Dataset class
    mnist_train_dataset = MNISTDataset(training_data_images, training_data_labels_one_hot) 
    mnist_test_dataset = MNISTDataset(testing_data_images, testing_data_labels_one_hot)

    # Print the shapes of the datasets
    print(f"Training dataset size: {len(mnist_train_dataset)}")  # Should be 60000
    print(f"Testing dataset size: {len(mnist_test_dataset)}")  # Should be 10000