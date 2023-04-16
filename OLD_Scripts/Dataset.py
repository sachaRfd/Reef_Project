import os
import torch
import cv2
import tifffile
import numpy as np
from torch.utils.data import Dataset
import rasterio
import rasterio.plot


class data_set(Dataset):
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

    Comments:
    1. The dataset is a list of all the files in the directory
    2. The max_shape is the maximum shape of all the images in the dataset
    3. Transform is currently only to Tensor
    4. The __getitem__ method returns the 3 bands of the image
    5. The find_max_shape runs through all the images --> Could be slow

    """

    def __init__(self, directory, transform=torch.Tensor):
        """
        Initialize the dataset

        Parameters
        ----------
        directory : str
            The directory of the data
        """
        self.directory = directory
        self.files = [os.path.join(directory, f) for f in os.listdir(
            directory) if f.endswith('.tif') or f.endswith('.tiff')]
        self.max_shape = self.find_max_shape()
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        data = tifffile.imread(path)  # Read the Tiff File --> Preserves memody compared to rasterio  + No need for geographic information anymore# noqa

        # convert the array of Uint8 to Float64 so transformation can be applied  # noqa
        data = data.astype(float)

        # Resize the image to maximum size of the dataset --> CV2 Keeps the aspect ratio of image intact:  # noqa
        data = cv2.resize(data, (self.max_shape[0], self.max_shape[1]))

        # Transform the image:
        if self.transform:
            data = self.transform(data)

        # Return the 3 bands
        return data[..., 0], data[..., 1], data[..., 2]

    def find_max_shape(self):
        max_shape = [0, 0]
        for f in self.files:
            data = tifffile.imread(f)
            shape = data.shape[:2]
            max_shape[0] = max(max_shape[0], shape[0])
            max_shape[1] = max(max_shape[1], shape[1])
        return tuple(max_shape)


if __name__ == "__main__":
    transform = torch.tensor
    data = data_set("Clipped_Reefs/Mini/", transform=transform)

    for i in range(10):
        band1, band2, band3 = data[i]
        print(band1.shape)
        print("Done")

    band1, band2, band3 = data[0]
    print("Band 1:", band1)
    print("Band 2:", band2)
    print("Band 3:", band3)

    # Tensors to numpy arrays
    band1_np = band1.numpy().astype(np.float64)
    band2_np = band2.numpy().astype(np.float64)
    band3_np = band3.numpy().astype(np.float64)

    # Visualise the image:
    with rasterio.open("test.tiff", "w", width=band1_np.shape[1],
                       height=band1_np.shape[0], count=3,
                       dtype=band1_np.dtype) as test:
        # First Channel
        test.write(band1_np, 1)
        # Second Channel
        test.write(band2_np, 2)
        # Third Channel
        test.write(band3_np, 3)
    with rasterio.open("test.tiff") as test:
        rasterio.plot.show(test)
