# visualise the different bands:
import rasterio
from rasterio.windows import Window
from rasterio.plot import show
import matplotlib.pyplot as plt
import numpy as np
from L21_RAE import RobustL21Autoencoder
import tensorflow.compat.v1 as tf  # noqa
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


# Run Code:
with rasterio.open("fake_timeseries_2/6478_T55LCD_20170825T004001no_transform.tiff") as src:  # noqa
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
    green_noise_low = green.copy()
    red_noise_low = red.copy()
    blue_noise_low = blue.copy()
    green_noise_high = green.copy()
    red_noise_high = red.copy()
    blue_noise_high = blue.copy()

    # Set Random Seed:
    np.random.seed(42)

    # Create two different masks for noise of low and noise of high:
    mask_low = np.random.choice([0, 0.5], size=green.shape, p=[0.95, 0.05])
    mask_high = np.random.choice([0, 0.5], size=green.shape, p=[0.80, 0.2])

    # Add random noise to the green band:
    green_noise_low += mask_low * \
        np.random.normal(0, 0.05, size=green_noise_low.shape)
    green_noise_high += mask_high * \
        np.random.normal(0, 0.3, size=green_noise_high.shape)
    red_noise_low += mask_low * \
        np.random.normal(0, 0.05, size=red_noise_low.shape)
    red_noise_high += mask_high * \
        np.random.normal(0, 0.3, size=red_noise_high.shape)
    blue_noise_low += mask_low * \
        np.random.normal(0, 0.05, size=blue_noise_low.shape)
    blue_noise_high += mask_high * \
        np.random.normal(0, 0.3, size=blue_noise_high.shape)

    # Add HARD NOISE:
    # green_noise_low += mask_low
    # green_noise_high += mask_high

# Run the Plotting Function:
green_noise_low_recon = create_full_image(green_noise_low, (28, 28))
red_noise_low_recon = create_full_image(red_noise_low, (28, 28))
blue_noise_low_recon = create_full_image(blue_noise_low, (28, 28))

green_noise_high_recon = create_full_image(green_noise_high, (28, 28))
red_noise_high_recon = create_full_image(red_noise_high, (28, 28))
blue_noise_high_recon = create_full_image(blue_noise_high, (28, 28))

true_recon_green = create_full_image(green, (28, 28))
true_recon_red = create_full_image(red, (28, 28))
true_recon_blue = create_full_image(blue, (28, 28))

print("MSE of the original image: ", mean_squared_error(green,
                                                        true_recon_green))
print("MSE of the low noise image: ", mean_squared_error(
    green_noise_low, green_noise_low_recon))
print("MSE of the high noise image: ", mean_squared_error(
    green_noise_high, green_noise_high_recon))

# Plot the original image, reconstructed image, noisy image next to it too:
# add scale bars to each graph:
fig, axs = plt.subplots(4, 3, figsize=(15, 10))
axs[0, 0].set_title("Original Green Band")
show(green, ax=axs[0, 0])
axs[0, 1].set_title("Low Noise Green Band")
show(green_noise_low, ax=axs[0, 1])
axs[0, 2].set_title("High Noise Green Band")
show(green_noise_high, ax=axs[0, 2])
# plot the noise mask:
axs[1, 0].set_title("Zero Noise Mask")
show(green - green, ax=axs[1, 0])
axs[1, 1].set_title("Low Noise Mask")
show(mask_low, ax=axs[1, 1])
axs[1, 2].set_title("High Noise Mask")
show(mask_high, ax=axs[1, 2])

# Plot the reconstructed images:
axs[2, 0].set_title("Original Image Reconstruction")
show(true_recon_green, ax=axs[2, 0])
axs[2, 1].set_title("Low Noise Image Reconstruction")
show(green_noise_low_recon, ax=axs[2, 1])
axs[2, 2].set_title("High Noise Image Reconstruction")
show(green_noise_high_recon, ax=axs[2, 2])

# plot the SE of the images as images:
axs[3, 0].set_title("SE of Original Image")
show(np.sqrt((green - true_recon_green)**2), ax=axs[3, 0])
axs[3, 1].set_title("SE of Low Noise Image")
show(np.sqrt((green_noise_low - green_noise_low_recon)**2),
     ax=axs[3, 1])
axs[3, 2].set_title("SE of High Noise Image")
show(np.sqrt((green_noise_high - green_noise_high_recon)**2),
     ax=axs[3, 2])
plt.tight_layout()
# save the figure:
plt.savefig("fake_timeseries/green_band_lambda_005.png")

# Now do the same but for the red band:
# Run the Plotting Function:
fig, axs = plt.subplots(4, 3, figsize=(15, 10))
axs[0, 0].set_title("Original Red Band")
show(red, ax=axs[0, 0])
axs[0, 1].set_title("Low Noise Red Band")
show(red_noise_low, ax=axs[0, 1])
axs[0, 2].set_title("High Noise Red Band")
show(red_noise_high, ax=axs[0, 2])
# plot the noise mask:
axs[1, 0].set_title("Zero Noise Mask")
show(red - red, ax=axs[1, 0])
axs[1, 1].set_title("Low Noise Mask")
show(mask_low, ax=axs[1, 1])
axs[1, 2].set_title("High Noise Mask")
show(mask_high, ax=axs[1, 2])

# Plot the reconstructed images:
axs[2, 0].set_title("Original Image Reconstruction")
show(true_recon_red, ax=axs[2, 0])
axs[2, 1].set_title("Low Noise Image Reconstruction")
show(red_noise_low_recon, ax=axs[2, 1])
axs[2, 2].set_title("High Noise Image Reconstruction")
show(red_noise_high_recon, ax=axs[2, 2])

# plot the SE of the images as images:
axs[3, 0].set_title("SE of Original Image")
show(np.sqrt((red - true_recon_red)**2), ax=axs[3, 0])
axs[3, 1].set_title("SE of Low Noise Image")
show(np.sqrt((red_noise_low - red_noise_low_recon)**2),
     ax=axs[3, 1])
axs[3, 2].set_title("SE of High Noise Image")
show(np.sqrt((red_noise_high - red_noise_high_recon)**2),
     ax=axs[3, 2])
plt.tight_layout()
# save the figure:
plt.savefig("fake_timeseries/red_band_mabda_005.png")

# Now do the same but for the Blue band:
# Run the Plotting Function:
fig, axs = plt.subplots(4, 3, figsize=(15, 10))
axs[0, 0].set_title("Original Blue Band")
show(blue, ax=axs[0, 0])
axs[0, 1].set_title("Low Noise Blue Band")
show(blue_noise_low, ax=axs[0, 1])
axs[0, 2].set_title("High Noise Blue Band")
show(blue_noise_high, ax=axs[0, 2])
# plot the noise mask:
axs[1, 0].set_title("Zero Noise Mask")
show(blue - blue, ax=axs[1, 0])
axs[1, 1].set_title("Low Noise Mask")
show(mask_low, ax=axs[1, 1])
axs[1, 2].set_title("High Noise Mask")
show(mask_high, ax=axs[1, 2])

# Plot the reconstructed images:
axs[2, 0].set_title("Original Image Reconstruction")
show(true_recon_blue, ax=axs[2, 0])
axs[2, 1].set_title("Low Noise Image Reconstruction")
show(blue_noise_low_recon, ax=axs[2, 1])
axs[2, 2].set_title("High Noise Image Reconstruction")
show(blue_noise_high_recon, ax=axs[2, 2])

# plot the SE of the images as images:
axs[3, 0].set_title("SE of Original Image")
show(np.sqrt((blue - true_recon_blue)**2), ax=axs[3, 0])
axs[3, 1].set_title("SE of Low Noise Image")
show(np.sqrt((blue_noise_low - blue_noise_low_recon)**2),
     ax=axs[3, 1])
axs[3, 2].set_title("SE of High Noise Image")
show(np.sqrt((blue_noise_high - blue_noise_high_recon)**2),
     ax=axs[3, 2])
plt.tight_layout()
# save the figure:
plt.savefig("fake_timeseries/blue_band_lambda_005.png")


# Now add all 3 reconstructions to a tiff file with rgb:

with rasterio.open("fake_timeseries/true_recon_lambda_005.tif", 'w',
                   driver='GTiff',
                   height=true_recon_green.shape[0],
                   width=true_recon_green.shape[1],
                   count=3,
                   dtype=true_recon_green.dtype,
                   crs=crs,
                   transform=transform) as dst:
    dst.write(true_recon_red, 1)
    dst.write(true_recon_green, 2)
    dst.write(true_recon_blue, 3)
