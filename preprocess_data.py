from mnist import MNIST
import torch

# Load the MNIST dataset
MNIST_dataset = MNIST('dataset')

# Define the maximum pixel value for normalization
MAX_PIXEL_VALUE = 255.0

# Define the one-hot encoding function for labels
def one_hot(label, num_classes=10):
    vec = torch.zeros(num_classes)  # Create a zero vector of length equal to the number of classes
    vec[label] = 1                  # Set the index corresponding to the label to 1
    return vec

# Define the preprocess_data function to load and preprocess the MNIST dataset
def preprocess_data():
    # Load training and testing data
    training_data_images, training_data_labels = MNIST_dataset.load_training()
    testing_data_images, testing_data_labels = MNIST_dataset.load_testing()

    # Convert data to torch tensors and normalize the pixel values to the range [0, 1]
    training_data_images = torch.tensor(training_data_images, dtype=torch.float32) / MAX_PIXEL_VALUE
    training_data_labels = torch.tensor(training_data_labels, dtype=torch.float32)

    # Convert testing data to torch tensors and normalize the pixel values to the range [0, 1]
    testing_data_images = torch.tensor(testing_data_images, dtype=torch.float32) / MAX_PIXEL_VALUE
    testing_data_labels = torch.tensor(testing_data_labels, dtype=torch.float32)

    # Convert training labels to one-hot encoding
    training_data_labels_one_hot = torch.zeros((training_data_labels.shape[0], 10))  # Create a tensor to hold one-hot encoded labels
    for i in range(training_data_labels.shape[0]):
        training_data_labels_one_hot[i] = one_hot(int(training_data_labels[i]))  # Convert each label to one-hot encoding   

    # Convert testing labels to one-hot encoding
    testing_data_labels_one_hot = torch.zeros((testing_data_labels.shape[0], 10))  # Create a tensor to hold one-hot encoded labels
    for i in range(testing_data_labels.shape[0]):
        testing_data_labels_one_hot[i] = one_hot(int(testing_data_labels[i]))  # Convert each label to one-hot encoding

    return training_data_images, training_data_labels_one_hot, testing_data_images, testing_data_labels_one_hot

if __name__ == "__main__":
    training_data_images, training_data_labels_one_hot, testing_data_images, testing_data_labels_one_hot = preprocess_data()

    # Print the shapes of the preprocessed data
    print(f"Training images shape: {training_data_images.shape}")  # Should be (60000, 784)
    print(f"Training labels shape (one-hot): {training_data_labels_one_hot.shape}")  # Should be (60000, 10)
    print(f"Testing images shape: {testing_data_images.shape}")  # Should be (10000, 784)
    print(f"Testing labels shape (one-hot): {testing_data_labels_one_hot.shape}")  # Should be (10000, 10)
    

