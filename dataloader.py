from torch.utils.data import Dataset

# Define the MNIST dataset class
class MNISTDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images 
        self.labels = labels 

    # Return the number of samples in the dataset
    def __len__(self):
        return len(self.images)

    # Return a single sample (image and label) from the dataset at the specified index
    def __getitem__(self, idx):
        image = self.images[idx].float()  # Normalize the image
        label = self.labels[idx]
        return image, label
    
# Example usage of the MNISTDataset class
if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from mnist import MNIST
    import torch

    # Define the maximum pixel value for normalization
    MAX_PIXEL_VALUE = 255.0

    # Load the MNIST dataset
    MNIST_dataset = MNIST('dataset')

    # Load training and testing data
    training_data_images, training_data_labels = MNIST_dataset.load_training()
    testing_data_images, testing_data_labels = MNIST_dataset.load_testing()

    # Convert data to torch tensors and normalize the pixel values to the range [0, 1]
    training_data_images = torch.tensor(training_data_images, dtype=torch.float32) / MAX_PIXEL_VALUE
    training_data_labels = torch.tensor(training_data_labels, dtype=torch.float32)

    # Convert testing data to torch tensors and normalize the pixel values to the range [0, 1]
    testing_data_images = torch.tensor(testing_data_images, dtype=torch.float32) / MAX_PIXEL_VALUE
    testing_data_labels = torch.tensor(testing_data_labels, dtype=torch.float32)

    # Create the MNIST dataset
    mnist_train_dataset = MNISTDataset(training_data_images, training_data_labels) 
    mnist_test_dataset = MNISTDataset(testing_data_images, testing_data_labels)

    # Example of accessing a sample from the dataset
    sample_image, sample_label = mnist_train_dataset[torch.randint(0, 1000, (1,))]  # Get a random sample from the training dataset

    # Print the shape of the sample image and the corresponding label
    print(f"Sample image shape: {sample_image.shape}")  # Should be (784,)
    print(f"Sample label: {sample_label}")  # Should be a scalar value representing the label of the image

    # Visualize the sample image using matplotlib
    plt.imshow(sample_image.reshape(28, 28), cmap='gray')
    plt.title(f"Label: {int(sample_label.item())}")
    plt.show()