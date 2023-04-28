import sys
import os
import tifffile
import numpy as np
import matplotlib.pyplot as plt


# Make a class that reads files from the directory and saves all the data as npy file in a directory: # noqa
class create_dataset():
    """
    Create a dataset from the directory

    Parameters
    ----------
    directory : str
        The directory of the data
    max_shape : int
        The maximum shape of the images in the dataset
    save_dir : str, optional
        The directory to save the dataset to, by default None

    Returns
    -------
    Dataset
        The dataset

    For Now I am testing it with the Mini Dataset --> which contains around 150 images  # noqa

    Now Includes Min-Max Normalisation
    Comments:
    1. The dataset is a list of all the files in the directory
    2. The max_shape is the maximum shape of all the images in the dataset
    3. Transform is currently only to Tensor
    4. The __getitem__ method returns the 3 bands of the image
    5. The find_max_shape runs through all the images --> Could be slow

    """

    def __init__(self, directory, max_shape, save_dir=None):
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

                # Loop to reshape to biggest Image
                # else:
                #     self.max_shape[0] = min(self.max_shape[0], shape[0])
                #     self.max_shape[1]

        # Save the dataset to a directory:
        if save_dir is not None:
            self.save_dataset(save_dir)

    def save_dataset(self, save_dir):
        """
        Save the dataset to a directory

        Parameters
        ----------
        save_dir : str
            The directory to save the dataset to

        Returns
        -------
        None
        """
        # Create the directory if it does not exist:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Loop through all the files and save them:
        for i, path in enumerate(self.files):
            data = tifffile.imread(path)

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
                # Save the image:
                # print(data.shape)
                np.save(os.path.join(save_dir, f'{i}.npy'), data)
                # Plot the image in dataset_test_images directory:
                plt.imsave(os.path.join(
                    'dataset_images_examples', f'{i}.png'), data)

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
                        # print(subimage.shape)
                        np.save(os.path.join(
                            save_dir, f'{i}_{j}.npy'), subimage)
                        plt.imsave(os.path.join(
                            'dataset_images_examples', f'{i}_{j}.png'), subimage)  # noqa
            # If the image is the same size as the shape we want --> Do nothing
            elif data.shape[0] == self.max_shape[0] and data.shape[1] == self.max_shape[1]:  # noqa
                # print(data.shape)
                np.save(os.path.join(save_dir, f'{i}.npy'), data)
                plt.imsave(os.path.join(
                    'dataset_images_examples', f'{i}.png'), data)
                pass


if __name__ == "__main__":  # Clipped_Reefs/clean/ 75 Reef_data_numpy/75_pix_reefs_PADDED # noqa
    if len(sys.argv) != 4:
        print("Usage: python script.py <Path/to/images>  <size of final padded images>  <Output Path>")  # noqa
        sys.exit(1)

    image_path, padded_images, output_path = str(
        sys.argv[1]), str(sys.argv[2]), str(sys.argv[3])

    root_path = str(sys.argv[1])
    # transform = torch.tensor
    data = create_dataset(image_path, int(padded_images), save_dir=output_path)  # noqa
    print("Dataset has been Loaded into : ", output_path)
