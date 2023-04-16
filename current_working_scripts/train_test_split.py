# Class that loads train and test sets:
import os
import tifffile
import numpy as np
import matplotlib.pyplot as plt


class dataset():
    """
    """
    def __init__(self, directory, max_shape):
        """
        Initialize the dataset

        Parameters
        ----------
        directory : str
            The directory of the data
        """
        self.directory = directory
        self.files = []
        self.max_shape = [max_shape, max_shape]
        for f in os.listdir(directory):
            if f.endswith('.tif') or f.endswith('.tiff'):
                path = os.path.join(directory, f)
                data = tifffile.imread(path)
                shape = data.shape[:2]
                # The first check checks that not all the values in the data are the same --> As this would mean that we have a corrupt image:  # noqa
                # It then checks for images are at least bigger than the max shape or smaller than them. If they are bigger than the max shape --> We will crop them into smaller images:  # noqa
                if np.any(data != data[0, 0]) and ((shape[0] < self.max_shape[0] and shape[1] < self.max_shape[1])  # noqa
                                                   or (shape[0] > self.max_shape[0] and shape[1] > self.max_shape[1])  # noqa
                                                   or (shape[0] == self.max_shape[0] and shape[1] == self.max_shape[1])):  # noqa
                    self.files.append(path)
        # Load the dataset to a directory:
        self.loaded_data = self.load_dataset()


    def load_dataset(self):
        """
        Load the data into a np array, ready for training
        """

        all_data = []  # Setup list

        # Loop through all the files and save them:
        for i, path in enumerate(self.files):
            data = tifffile.imread(path)  # Read the image at path
            print(path)  # print path for testing

            # Min Max Normalisation:
            data = (data - np.min(data)) / \
                ((np.max(data) - np.min(data)+1e-10))
            
            # Get average of the 3 bands:
            data = (data[:, :, 0] + data[:, :, 1] + data[:, :, 2]) / 3

            # If the image is smaller than the shape we want --> Add padding to it  # noqa
            if data.shape[0] < self.max_shape[0] and data.shape[1] < self.max_shape[1]:  # noqa
                # Find the padding we need to add:
                padding = [self.max_shape[0] - data.shape[0],
                           self.max_shape[1] - data.shape[1]]
                # Add the padding:
                data = np.pad(data, ((0, padding[0]), (0, padding[1])),
                              mode='constant', constant_values=0)

                # Append Image:
                all_data.append(data)

            # If the image is bigger than the shape we want --> Crop into smaller subimages in a for loop:  # noqa
            elif data.shape[0] > self.max_shape[0] and data.shape[1] > self.max_shape[1]:  # noqa
                # Find the number of subimages we can make:
                num_subimages = (
                    data.shape[0] // self.max_shape[0]) * (data.shape[1] // self.max_shape[1])  # noqa
                # Loop through all the subimages and save them:
                for j in range(num_subimages):
                    # Find the starting and ending indices:
                    start_x = (j // (data.shape[1] // self.max_shape[1])) * \
                        self.max_shape[0]
                    end_x = start_x + self.max_shape[0]
                    start_y = (j % (data.shape[1] // self.max_shape[1])) * \
                        self.max_shape[1]
                    end_y = start_y + self.max_shape[1]
                    # Get the subimage:
                    subimage = data[start_x:end_x, start_y:end_y]

                    # Save the subimage if it has 4 different pixel values:
                    if len(np.unique(subimage)) > 4:

                        # Append Sub-Image:
                        all_data.append(subimage)

            # If the image is the same size as the shape we want --> Do nothing
            elif data.shape[0] == self.max_shape[0] and data.shape[1] == self.max_shape[1]:  # noqa
                # Append Image:
                all_data.append(data)
                pass

        # Reshape the data into (num_images, height*width):
        all_data = np.array(all_data).reshape(
            len(all_data), self.max_shape[0]*self.max_shape[1])
        
        # Shuffle the data for correct training:
        np.random.seed(76)  # Set seed to be able to reproduce
        np.random.shuffle(all_data)
        return all_data  # Return the data

    def return_data(self):
        '''
        Function that returns a train / test splitted, shuffled data
        '''
        # Split the data into training and testing set where test is 10% of the data:  # noqa
        train_set = self.loaded_data[:int(0.9*len(self.loaded_data))]
        test_set = self.loaded_data[int(0.9*len(self.loaded_data)):]
        return train_set, test_set


if __name__ == "__main__":  # Clipped_Reefs/clean/ 75 Reef_data_numpy/75_pix_reefs_PADDED # noqa

    image_path = "Clipped_Reefs/no_clouds/"  # Read all the clipped reefs
    padded_images = 28  # Image size to create
    dataset = dataset(image_path, int(padded_images))  # Setup the dataset  # noqa
    X_train, X_test = dataset.return_data()  # Get the Array for training

    # # save the first 500 samples to visuale data: 
    # for i in range(10):  # Use to be the whole dataset size
    #     fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    #     band_image = X_train[i]
    #     # band_image = band_image.numpy().astype(np.float32)
    #     band_image = band_image.reshape(int(padded_images), int(padded_images))
    #     ax.imshow(band_image)
    #     plt.savefig("visual_data/" + f"{int(padded_images)}_images_test_{i}.png")
    #     plt.close(fig)


    print("Shape of the train set is : ", X_train.shape)
    print("Shape of the test set is : ", X_test.shape)
