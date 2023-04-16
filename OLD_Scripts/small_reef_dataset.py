import sys
import os
import torch
import cv2
import tifffile
import numpy as np
from torch.utils.data import Dataset
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
                if shape[0] <= self.max_shape[0] and shape[1] <= self.max_shape[1]:  # noqa
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

        if data.shape[0] < self.max_shape[0] or data.shape[1] < self.max_shape[1]:  # noqa
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

        if data.shape[0] > self.max_shape[0] or data.shape[1] > self.max_shape[1]:  # noqa
            return None

        # convert the array of Uint8 to Float64 so transformation can be applied  # noqa
        data = data.astype(float)

        # Transform the image:
        if self.transform:
            data = self.transform(data)

        # Return the mean of the 3 bands
        data_mean = (data[..., 0] + data[..., 1] + data[..., 2]) / 3

        return data_mean


if __name__ == "__main__":  # Clipped_Reefs/clean/ 75 Reef_data_numpy/75_pix_reefs_PADDED # noqa
    if len(sys.argv) != 4:
        print("Usage: python script.py <Path/to/images>  <size of final padded images>  <Output Path>")  # noqa
        sys.exit(1)

    image_path, padded_images, output_path = str(
        sys.argv[1]), str(sys.argv[2]), str(sys.argv[3])

    root_path = str(sys.argv[1])

    transform = torch.tensor
    data = small_reef_dataset(image_path, int(padded_images), transform=transform)  # noqa

    # Get Basic band to copy shape and file:
    band_average = data[5]
    print("Band average:", band_average)
    print(band_average.shape)

    # # Tensors to numpy arrays
    band_average_np = band_average.numpy().astype(np.float32)

    # Create .npz file with all the data from the reefs for use with tensorflow in low size:  # noqa
    for i in range(data.__len__()):
        band_average = data[i]
        band_average_np = band_average.numpy().astype(np.float32)
        np.save(output_path + f"/Reef_{i}", band_average_np)  # noqa
        print("Done with Reef_{}".format(i))
    print("Loaded NPY file")
