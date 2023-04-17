from rasterio.windows import Window
from L21_RAE import RobustL21Autoencoder
import tensorflow.compat.v1 as tf  # noqa
import numpy as np
import matplotlib.pyplot as plt
import os
import rasterio
tf.compat.v1.disable_eager_execution()


def run_model(X):
    # create a new instance of the RobustL21Autoencoder class
    sess = tf.Session()
    rae = RobustL21Autoencoder(sess=sess, lambda_=0.0001,
                               layers_sizes=[28*28, 400, 200])
    # load the saved weights
    saver = tf.train.Saver()
    saver.restore(sess, "saved_short_model/model")

    second_array = []
    for sub_array in X:
        # Run array through model:
        sub_array = sub_array.reshape(1, 784)
        reconstructed_data = rae.getRecon(sub_array, sess=sess)
        second_array.append(reconstructed_data)

    # Reshape second array to the same size as X:
    second_array = np.array(second_array)
    # Reshape to the same size as X:
    second_array = second_array.reshape(X.shape)
    return second_array


def generate_images(filepath, noise_level=0.3):
    """
    Generate images from model and save them to a file.
    """
    # Find image that ends with tiff: and save its name (not including the .tiff)  # noqa
    for file in os.listdir(filepath):
        if file.endswith(".tiff"):
            image = file
            file_name = file[:-5]
            print(f"File used: {file_name}")

    # Define the size of sub-images
    subimg_size = (28, 28)

    # Open the image using rasterio
    with rasterio.open(filepath + image) as dataset:
        # Read the image as a numpy array
        # Was previously reading only band 1 --> which is red
        image = dataset.read(2)

        # Get the size of the image
        height, width = image.shape
        # Pad the image so that its size is a multiple of the sub-image size
        pad_height = subimg_size[0] - (height % subimg_size[0])
        pad_width = subimg_size[1] - (width % subimg_size[1])
        image = np.pad(
            image, ((0, pad_height), (0, pad_width)), mode="constant")

        # Get the new size of the padded image
        height, width = image.shape

        # Define the number of rows and columns of sub-images
        rows = height // subimg_size[0]
        cols = width // subimg_size[1]

        # Create a new array to store the sub-images
        subimages = np.zeros(
            (rows, cols, subimg_size[0], subimg_size[1]), dtype=image.dtype)

        # Loop over the sub-images
        for r in range(rows):
            for c in range(cols):
                # Define the window for the sub-image
                window = Window(
                    c*subimg_size[1], r*subimg_size[0], subimg_size[1],
                    subimg_size[0])
                # Read the sub-image
                subimg = image[window.row_off:window.row_off +
                               window.height, window.col_off:window.col_off +
                               window.width]
                # Save the sub-image
                subimages[r, c] = subimg

        # Reshape the sub-images into a 2D array
        subimages_2d = subimages.reshape(
            rows*cols, subimg_size[0]*subimg_size[1])

        # Do a Min-Max Normalization of the whole sub-images array
        subimages_2d = (subimages_2d - subimages_2d.min()) / \
            ((subimages_2d.max() - subimages_2d.min() + 1e-10))

        # Use the model to reconstruct the sub-images
        reconstructed_2d = run_model(subimages_2d)

        # Reshape the reconstructed sub-images into a 4D array
        reconstructed = reconstructed_2d.reshape(
            rows, cols, subimg_size[0], subimg_size[1])

        # Create a new array to store the reconstructed image
        reconstructed_image = np.zeros((height, width), dtype=np.float32)

        # Loop over the sub-images and insert them into the reconstructed image
        for r in range(rows):
            for c in range(cols):
                subimg = reconstructed[r, c]
                reconstructed_image[r*subimg_size[0]:(
                    r+1)*subimg_size[0],
                    c*subimg_size[1]:(c+1)*subimg_size[1]] = subimg

        # Remove the padding from the reconstructed image
        reconstructed_image = reconstructed_image[:height -
                                                  pad_height, :width-pad_width]

        # Save the Reconstruction image as a tiff file with the same
        # geotransform as the original image:
        with rasterio.open((filepath + file_name + "_reconstructed_band2"),
                           'w',
                           driver='GTiff',
                           height=reconstructed_image.shape[0],
                           width=reconstructed_image.shape[1],
                           count=1, dtype=np.float32,
                           crs=dataset.crs,
                           transform=dataset.transform) as dst:

            dst.write(reconstructed_image, 1)

        # Remove the padding from the original image:
        image = image[:height - pad_height, :width-pad_width]

        # Plot the original image, reconstructed image, and difference image
        fig, axs = plt.subplots(1, 2, figsize=(20, 10))
        axs[0].imshow(image)
        axs[0].set_title("Original Image")
        axs[1].imshow(reconstructed_image)
        axs[1].set_title("Reconstructed Image")
        # Save the original and reconstructed images:
        plt.savefig(filepath + file_name + "_original_reconstructed_band2.png")

        # Plot the difference image:
        fig, axs = plt.subplots(1, 2, figsize=(20, 10))
        axs[0].imshow(image)
        axs[0].set_title("Original Image")
        axs[1].imshow(reconstructed_image - image)
        axs[1].set_title("Difference Image")
        # save the difference image:
        plt.savefig(filepath + file_name + "_difference_image_band2.png")

        # Plot the MSE image:
        fig, axs = plt.subplots(1, 2, figsize=(20, 10))
        axs[0].imshow(image)
        axs[0].set_title("Original Image")
        axs[1].imshow((reconstructed_image - image)**2)
        axs[1].set_title("MSE Image")
        # save the MSE image:
        plt.savefig(filepath + file_name + "_mse_image_band2.png")


# Run the Plotting Function:
generate_images("fake_timeseries/")
