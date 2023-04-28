# File that reads all the tiff files in the directory, puts them through model, and then plots them side by side:  # noqa
import rasterio
from rasterio.windows import Window
from rasterio.plot import show_hist  # noqa
import matplotlib.pyplot as plt
import numpy as np
from L21_RAE import RobustL21Autoencoder
import tensorflow.compat.v1 as tf  # noqa
import glob
from sklearn.metrics import mean_squared_error
tf.compat.v1.disable_eager_execution()


def run_model(X):
    # Clear the default graph - so that we can run the model multiple times:
    tf.reset_default_graph()

    # create a new instance of the RobustL21Autoencoder class
    sess = tf.Session()
    rae = RobustL21Autoencoder(sess=sess, lambda_=0.0001,
                               layers_sizes=[28*28, 400, 200])
    # load the saved weights
    saver = tf.train.Saver()
    # saver.restore(sess, "saved_short_model/model")
    saver.restore(sess, "Model_weights/short_model/005 lambda/model")

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


def create_full_image(image, subimg_size):
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

    # Remove the padding from the original image:
    image = image[:height - pad_height, :width-pad_width]

    return reconstructed_image


# Code Running:
# Loop over the fake_timeseries_2/ directory and get list of all the Tiff files:  # noqa
# Get list of all the tiff files in the directory:
tiff_files = glob.glob("fake_timeseries_4/*.tiff")

green_bands = []
blue_bands = []
red_bands = []

recon_green_bands = []
recon_blue_bands = []
recon_red_bands = []

data_list = []

counter = 0
for file in tiff_files:

    # Append the data of the file name to the data_list:
    data_list.append(file[30:38])

    with rasterio.open(file) as src:  # noqa
        # Get the Green band as np.array:
        green = np.array(src.read(3))
        red = np.array(src.read(4))
        blue = np.array(src.read(2))

        # get crs and transform:
        crs = src.crs
        transform = src.transform
        width, height = green.shape

        # Min-max scale the green band:
        green = (green - green.min()) / ((green.max() - green.min() + 1e-10))
        red = (red - red.min()) / ((red.max() - red.min() + 1e-10))
        blue = (blue - blue.min()) / ((blue.max() - blue.min() + 1e-10))

        # Get the Recognition images from model:
        true_recon_green = create_full_image(green, (28, 28))
        true_recon_red = create_full_image(red, (28, 28))
        true_recon_blue = create_full_image(blue, (28, 28))

        # Add the bands to the list for later plotting
        recon_green_bands.append(true_recon_green)
        recon_red_bands.append(true_recon_red)
        recon_blue_bands.append(true_recon_blue)

        # Add the original bands for later plotting:
        green_bands.append(green)
        red_bands.append(red)
        blue_bands.append(blue)

        # Do Loading bar:
        counter += 1
        print("Done with ", counter, "|", len(tiff_files))

# Now we can plot all the blue bands with the original next to it: with titles and scale bars:  # noqa

# Cumulative difference:
cummu_difference = np.zeros((green_bands[0].shape))

for i in range(len(blue_bands)):
    fig, ax = plt.subplots(1, 3, figsize=(10, 10))
    ax[0].imshow(blue_bands[i], cmap="Blues")
    ax[0].set_title("Original")
    ax[1].imshow(recon_blue_bands[i], cmap="Blues")
    ax[1].set_title("Reconstructed")
    ax[2].imshow(
        np.sqrt((blue_bands[i] - recon_blue_bands[i])**2), cmap="Blues")
    ax[2].set_title(
        f"SE Difference with MSE: {mean_squared_error(blue_bands[i], recon_blue_bands[i]):.6f}")  # noqa
    # Add the current cummu difference:  # noqa
    # cummu_difference += np.array(np.sqrt((blue_bands[i] - recon_blue_bands[i])**2))  # noqa
    # ax[3].imshow((cummu_difference), cmap="Blues")
    # ax[3].set_title("Cumulative Difference ")  # noqa
    fig.savefig("fake_timeseries_4/Blue_bands_{}.png".format(data_list[i]))
    plt.close()

# Now we can plot all the green bands with the original next to it:

# Cumulative difference:
cummu_difference = np.zeros((green_bands[0].shape))
for i in range(len(green_bands)):
    fig, ax = plt.subplots(1, 3, figsize=(10, 10))
    ax[0].imshow(green_bands[i], cmap="Greens")
    ax[0].set_title("Original")
    ax[1].imshow(recon_green_bands[i], cmap="Greens")
    ax[1].set_title("Reconstructed")
    ax[2].imshow(
        np.sqrt((green_bands[i] - recon_green_bands[i])**2), cmap="Greens")
    ax[2].set_title(
        f"SE Difference with MSE: {mean_squared_error(green_bands[i], recon_green_bands[i]):.6f}")  # noqa
    # Add the current cummu difference:  # noqa
    # cummu_difference += np.array(np.sqrt((green_bands[i] - recon_green_bands[i])**2))  # noqa
    # ax[3].imshow((cummu_difference), cmap="Greens")
    # ax[3].set_title("Cumulative Difference ")  # noqa

    fig.savefig("fake_timeseries_4/Green_bands_{}.png".format(data_list[i]))
    plt.close()

# Now we can plot all the red bands with the original next to it:
cummu_difference = np.zeros((green_bands[0].shape))
for i in range(len(red_bands)):
    fig, ax = plt.subplots(1, 3, figsize=(10, 10))
    ax[0].imshow(red_bands[i], cmap="Reds")
    ax[0].set_title("Original")
    ax[1].imshow(recon_red_bands[i], cmap="Reds")
    ax[1].set_title("Reconstructed")
    ax[2].imshow(np.sqrt((red_bands[i] - recon_red_bands[i])**2), cmap="Reds")
    ax[2].set_title(
        f"SE Difference with MSE: {mean_squared_error(red_bands[i], recon_red_bands[i]):.6f}")  # noqa
    # Add the current cummu difference:  # noqa
    # cummu_difference += np.array(np.sqrt((red_bands[i] - recon_red_bands[i])**2))  # noqa
    # ax[3].imshow((cummu_difference), cmap="Reds")
    # ax[3].set_title("Cumulative Difference ")  # noqa
    fig.savefig("fake_timeseries_4/Red_bands_{}.png".format(data_list[i]))
    plt.close()

# Now we can plot all the bands together to visualise the image in RGB:
cummu_difference = np.zeros((green_bands[0].shape))
for i in range(len(red_bands)):
    fig, ax = plt.subplots(1, 3, figsize=(10, 10))
    ax[0].imshow(np.dstack((red_bands[i], green_bands[i], blue_bands[i])))
    ax[0].set_title("Original")
    ax[1].imshow(np.dstack(
        (recon_red_bands[i], recon_green_bands[i], recon_blue_bands[i])))
    ax[1].set_title("Reconstructed")
    ax[2].imshow(np.sqrt((red_bands[i] - recon_red_bands[i])**2) +
                 np.sqrt((green_bands[i] - recon_green_bands[i])**2) +
                 np.sqrt((blue_bands[i] - recon_blue_bands[i])**2))
    ax[2].set_title(
        f"SE Difference with MSE: {mean_squared_error(red_bands[i], recon_red_bands[i]) + mean_squared_error(green_bands[i], recon_green_bands[i]) + mean_squared_error(blue_bands[i], recon_blue_bands[i]):.6f}")  # noqa
    # Add the current cummu difference:  # noqa
    # cummu_difference += np.array(np.sqrt((red_bands[i] - recon_red_bands[i])**2) +  # noqa
    #                              np.sqrt((green_bands[i] - recon_green_bands[i])**2) +  # noqa
    #                              np.sqrt((blue_bands[i] - recon_blue_bands[i])**2))  # noqa
    # ax[3].imshow((cummu_difference), cmap="Greys")
    # ax[3].set_title("Cumulative Difference ")  # noqa
    fig.savefig("fake_timeseries_4/RGB_bands_{}.png".format(data_list[i]))
    plt.close()


# Make the TIFF rasterio files with 3 counts:
# for i in range(len(red_bands)):
#     # Create the RGB image:
#     with rasterio.open("fake_timeseries_2/RGB_bands_{}.tif".format(data_list[i]), "w", driver="GTiff", height=height, width=width, count=3, dtype=np.float32, crs=crs, transform=transform) as dst:  # noqa
#         dst.write(recon_red_bands[i], 1)
#         dst.write(recon_green_bands[i], 2)
#         dst.write(recon_blue_bands[i], 3)

#     # Plot the Rasterio Histogram:
#     with rasterio.open("fake_timeseries_2/RGB_bands_{}.tif".format(data_list[i])) as src:  # noqa
#         fig, ax = plt.subplots(1, 1, figsize=(10, 10))
#         show_hist(src, bins=50, lw=0.0, stacked=False, alpha=0.3,
#                   histtype='stepfilled', title="Histogram", ax=ax)
#         fig.savefig(
#             "fake_timeseries_2/RGB_bands_{}_hist.png".format(data_list[i]))
#         plt.close()
