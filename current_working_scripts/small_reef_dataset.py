import sys
import os
import torch
import cv2
import tifffile
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import rasterio
import rasterio.plot  # noqa


class small_reef_dataset(Dataset):
    """
    Dataset class for the data

    Parameters
    ----------
    directory : str
        The directory of the data
    transform : torch.tensor, optional
        The transformation to apply to the data, by default None

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

    def __init__(self, directory, max_shape,  transform=torch.Tensor):
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
                # The first check checks that not all the values in the data are the same --> As this would mean that we have a corrupt image:
                # It then checks for images are at least bigger than the max shape or smaller than them.
                if np.any(data != data[0, 0]) and ((shape[0] < self.max_shape[0] and shape[1] < self.max_shape[1])
                                                    or (shape[0] > self.max_shape[0] and shape[1] > self.max_shape[1])
                                                    or (shape[0] == self.max_shape[0] and shape[1] == self.max_shape[1])):
                    self.files.append(path)

                # Loop to reshape to biggest Image
                # else:
                #     self.max_shape[0] = min(self.max_shape[0], shape[0])
                #     self.max_shape[1] = min(self.max_shape[1], shape[1])
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        data = tifffile.imread(path)

        # If the image is smaller than the shape we want --> Add padding to it
        if data.shape[0] < self.max_shape[0] and data.shape[1] < self.max_shape[1]:  # noqa
            print("Original Shape: ", data.shape)

            print(path)
            # Min Max Normalise the data:
            data = (data - np.min(data)) / ((np.max(data) - np.min(data)+1e-8))

            # Resize the image to 50 by 50 using CV2
            # data = cv2.resize(data, (self.max_shape[0], self.max_shape[1]))

            # Pad the image with zeros instead of resizing:
            # Calculate the amount of padding needed on each side
            pad_top = (self.max_shape[0] - data.shape[0]) // 2
            pad_bottom = self.max_shape[0] - data.shape[0] - pad_top
            pad_left = (self.max_shape[1] - data.shape[1]) // 2
            pad_right = self.max_shape[1] - data.shape[1] - pad_left
            # Pad the image using copyMakeBorder()
            data = cv2.copyMakeBorder(data.copy(),
                                      pad_top, pad_bottom,
                                      pad_left, pad_right,
                                      cv2.BORDER_CONSTANT, value=0)

            # Print the new shape
            print("New Shape: ", data.shape)

        # If the shape is larger than we want --> Create sub-images
        elif data.shape[0] > self.max_shape[0] and data.shape[1] > self.max_shape[1]:  # noqa
            
            # Normalise the data: 
            data = (data - np.min(data)) / ((np.max(data) - np.min(data)+1e-8))

            # Splitting the larger reefs into multiple small ones
            list_of_smaller_reefs = []
            for i in range(0, data.shape[0], self.max_shape[0]):
                for j in range(0, data.shape[1], self.max_shape[1]):
                    # Get coordinate of sub-image:
                    x = j
                    y = i
                    w = self.max_shape[0]
                    h = self.max_shape[1]  # add one for indexing
                    # Crop the images: 
                    list_of_smaller_reefs.append(data[y:y+h, x:x+w])  # append the smaller reefs
            
            # Set data to be the first image
            data = list_of_smaller_reefs[0]
            # Then set to the next imamge is if it isnt all black or white
            if len(list_of_smaller_reefs) > 3:
                if list_of_smaller_reefs[1].shape[0] == self.max_shape[0] and list_of_smaller_reefs[1].shape[1] == self.max_shape[1]:
                    data = list_of_smaller_reefs[1] 
            # If there are other images that are not all black or white then make that image the one given out:
            # for image in list_of_smaller_reefs:
            #     # image = np.array(image)
            #     if np.any(image != image[0, 0]):
            #         data = image
            #         break
            # For now make it so the data becomes one of these smaller reefs
            # data = list_of_smaller_reefs[0]
            print(f"New shape after using smaller section of the reef is: {data.shape}")

        # If the images are the right size --> Just need to normalise the data
        elif data.shape[0] == self.max_shape[0] and data.shape[1] == self.max_shape[1]:  # noqa
            # Normalise the data: 
            data = (data - np.min(data)) / ((np.max(data) - np.min(data)+1e-8))

        # This is just used for testing:
        else:
            print("There is a mistmatch")

        # convert the array of Uint8 to Float64 so transformation can be applied  # noqa
        data = data.astype(float)

        # Transform the image:
        if self.transform:
            data = self.transform(data)

        # Return the mean of the 3 bands
        # data_mean = (data[..., 0] + data[..., 1] + data[..., 2]) / 3

        # return the Green Band only: which is the second one: 
        data_green = data[..., 0]

        print(path)
        return data_green


if __name__ == "__main__":  # Clipped_Reefs/clean/ 75 Reef_data_numpy/75_pix_reefs_PADDED # noqa
    if len(sys.argv) != 4:
        print("Usage: python script.py <Path/to/images>  <size of final padded images>  <Output Path>")  # noqa
        sys.exit(1)

    image_path, padded_images, output_path = str(
        sys.argv[1]), str(sys.argv[2]), str(sys.argv[3])

    root_path = str(sys.argv[1])

    # transform = torch.tensor
    data = small_reef_dataset(image_path, int(padded_images), transform=None)  # noqa

    # Following Loop is to be able to visualise some of the images:
    for i in range(10):  # Use to be the whole dataset size
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        band_image = data[i]
        # band_image = band_image.numpy().astype(np.float32)
        band_image.reshape(int(padded_images), int(padded_images))
        ax.imshow(band_image)
        plt.savefig("visual_dataset_0_band/" + f"{int(padded_images)}_images_test_{i}.png")
        plt.close(fig)

    # Create .npz file with all the data from the reefs for use with tensorflow in low size:  # noqa
    # for i in range(data.__len__()):
    #     band_average = data[i]
    #     band_average_np = band_average.numpy().astype(np.float32)
    #     print(band_average_np)
    #     np.save(output_path + f"/Reef_{i}", band_average_np)  # noqa
    #     print("Done with Reef_{}".format(i))
    # print("Loaded NPY file")
