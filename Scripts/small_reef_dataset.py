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
        self.files = []
        self.max_shape = [300, 300]
        for f in os.listdir(directory):
            if f.endswith('.tif') or f.endswith('.tiff'):
                path = os.path.join(directory, f)
                data = tifffile.imread(path)
                shape = data.shape[:2]
                if shape[0] <= 300 and shape[1] <= 300:
                    self.files.append(path)
                else:
                    self.max_shape[0] = min(self.max_shape[0], shape[0])
                    self.max_shape[1] = min(self.max_shape[1], shape[1])
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        data = tifffile.imread(path)

        if data.shape[0] < 300 or data.shape[1] < 300:
            data = cv2.resize(data, (self.max_shape[1], self.max_shape[0]))

        if data.shape[0] > 300 or data.shape[1] > 300:
            return None

        # convert the array of Uint8 to Float64 so transformation can be applied  # noqa
        data = data.astype(float)

        # Resize the image to 300 by 300 using CV2
        data = cv2.resize(data, (300, 300))

        # Transform the image:
        if self.transform:
            data = self.transform(data)

        # Return the mean of the 3 bands
        data_mean = (data[..., 0] + data[..., 1] + data[..., 2]) / 3
        return data_mean


if __name__ == "__main__":
    transform = torch.tensor
    data = small_reef_dataset("Clipped_Reefs/all/", transform=transform)

    band_average = data[0]
    print("Band average:", band_average)
    print(band_average.shape)

    # # Tensors to numpy arrays
    # band_average_np = band_average.numpy().astype(np.float32)

    # # Visualise the image:
    # with rasterio.open("test_Average.tiff", "w", width=band_average_np.shape[1],  # noqa
    #                    height=band_average_np.shape[0], count=1,
    #                    dtype=band_average_np.dtype) as test:
    #     # Average Channel
    #     test.write(band_average_np, 1)
    # with rasterio.open("test_Average.tiff") as test:
    #     rasterio.plot.show(test)

    # Create .npz file with all the data from the reefs for use with tensorflow in low size:  # noqa
    for i in range(data.__len__()):
        band_average = data[i]
        band_average_np = band_average.numpy().astype(np.float32)
        np.save(f"Reef_data_numpy/small_reefs/Reef_{i}", band_average_np)
        print("Done with Reef_{}".format(i))
    print("Loaded NPY file")

    # for i in range(10):
    #     band_average = data[i]
    #     band_average_np = band_average.numpy().astype(np.uint8)
    #     np.save("Reef_data_numpy/Reef_{}".format(i), band_average_np)
    #     print("Done with Reef_{}".format(i))

    # for i in range(10):
    #     band_average = data[i]
    #     band_average_np = band_average.numpy().astype(np.float32)
    #     np.save("Reef_data_numpy/Reef_{}".format(i), band_average_np)
    #     print("Done with Reef_{}".format(i))

    # test = np.load("Reef_data_numpy/Reef_0.npy")
    # print(test)
