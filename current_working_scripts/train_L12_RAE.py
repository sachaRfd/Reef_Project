# Train File for the L21 RAE
import sys
import os
from L21_RAE import *  # noqa   # To reverse remove _dropout
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from train_test_split import *

# Show if the GPU is present:
print(tf.config.list_physical_devices('GPU'))

# Function to run L21 Model:
def run_model(data, test_data, input_size, lambda_, save_directory):
    # Check if GPU is available
    if tf.config.list_physical_devices('GPU'):
        # Specify GPU device
        with tf.device('/GPU:0'):
            inner_iteration = 10  
            iteration = 200
            with tf.compat.v1.Session() as sess:
                rae = RobustL21Autoencoder(sess=sess, lambda_=lambda_, layers_sizes=[input_size*input_size, 400, 200])  # 2_000, 1_500, 800])  # noqa
                L, S, list_of_costs, list_of_costs_testing = rae.fit(data, test_data, sess=sess, learning_rate=.0005, batch_size=200, inner_iteration=inner_iteration,  # noqa
                               iteration=iteration, verbose=True, save_directory=save_directory)  # noqa

                # Setup convergence plots:
                list_of_costs = np.array(list_of_costs)
                list_of_costs_testing = np.array(list_of_costs_testing)
                print(
                    f"Shape of the list of the costs of training: {list_of_costs.shape}")  # noqa
                print(
                    f"Shape of the list of the costs of testing: {list_of_costs_testing.shape}")  # noqa

                # Plot the Convergence plot: 
                for i in range(iteration):
                    fig, ax = plt.subplots(1, figsize=(10, 5))
                    ax.plot(range(0, inner_iteration), list_of_costs[i],color="blue", label="Training Loss" )
                    ax.plot(range(0, inner_iteration), list_of_costs_testing[i], color = "orange", label="Testing Loss")
                    ax.set_title(f"Convergence plot for the {i}'th iteration of the Robust Autoencoder")
                    ax.legend()  # Show the labels
                    plt.savefig((save_directory + f"Convergence_plot_{i}.png"))
                    plt.close(fig)

                # Merge the lists of costs across all iterations
                list_of_costs_all = np.ravel(list_of_costs)  # Unravel the vector
                list_of_costs_testing_all = np.ravel(list_of_costs_testing)  # Unravel the vector


                # Plot the whole Convergence plot: 
                fig, ax = plt.subplots(1, figsize=(10, 5))
                ax.plot(range(0, iteration*inner_iteration), list_of_costs_all, color="blue", label="Training Loss" )
                ax.plot(range(0, iteration*inner_iteration), list_of_costs_testing_all, color="orange", label="Testing Loss")
                ax.set_title("Convergence plot for all iterations of the Robust Autoencoder")
                ax.legend()  # Show the labels
                plt.savefig((save_directory + "Full_Convergence_plot.png"))
                plt.close(fig)

                # Get the recognition from the train data:
                recon_train = rae.getRecon(data, sess=sess)

                # Save some examples as PNG:
                for i in range(10):
                    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                    ax[0].imshow(data[i].reshape(input_size, input_size))
                    ax[0].set_title("Original Image")
                    ax[1].imshow(recon_train[i].reshape(
                        input_size, input_size))
                    ax[1].set_title("Reconstructed Image from training")
                    plt.savefig(
                        (save_directory + f"_training_example_{i}.png"))
                    plt.close(fig)

                # Plot some example L and S matrices from the final training process
                L = L.reshape(data.shape[0], input_size, input_size)
                S = S.reshape(data.shape[0], input_size, input_size)
                for i in range(5):
                    # Plot the L and S matrices to see what they have captured
                    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                    ax[0].imshow(L[i], cmap='gray')
                    ax[0].set_title("Low Rank (L) Matrix")
                    ax[1].imshow(S[i], cmap='gray')
                    ax[1].set_title("Sparse (S) Matrix")
                    plt.savefig((save_directory + f"Final_Matrices{i}.png"))
                    plt.close(fig)

                # Get the recognition using the test data:
                recon_test = rae.getRecon(test_data, sess=sess)

                # Save some examples as PNG:
                for i in range(5):
                    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                    ax[0].imshow(test_data[i].reshape(input_size, input_size))
                    ax[0].set_title("Original Image")
                    ax[1].imshow(recon_test[i].reshape(input_size, input_size))
                    ax[1].set_title("Reconstructed Image")
                    plt.savefig((save_directory + f"example_{i}.png"))
                    plt.close(fig)
                # Save some examples as TIFF files:
                for i in range(5):
                    # Create Tiff File
                    file_name = f"_test_image_{i}"
                    image = rasterio.open(save_directory + file_name + ".tiff", 'w', driver='Gtiff',  # noqa
                                          width=input_size, height=input_size,  # noqa
                                          count=1,
                                          crs=None,
                                          transform=None,
                                          dtype=np.float32
                                          )
                    image.write(recon_test[i].reshape(
                        input_size, input_size), 1)
                    image.close()


                # Save Model: Create a saver object
                saver = tf.compat.v1.train.Saver()
                # 'Model_Checkpoints/training_1/model')
                saver.save(sess, save_directory + "model")
                print("Model saved in path: %s" % save_directory)
    else:
        print("No GPU available - Please try again")


if __name__ == "__main__":
    # If statement to check that it is the right size:
    if len(sys.argv) != 5:
        # print("Usage: python script.py <Path/to/training files> <num_training_images> <path/to/testing/images> <num_test_images> <SIZE of input image SIZExSIZE> <Path/to/Save Directory>")  # noqa
        print("Usage: python script.py <Path to Clipped Images to resize> <Size of Images> <Lambda Value> <Path to Save>")
        sys.exit(1)

    # Get variables from command line:
    clipped_reefs_path, image_size, lambda_, save_directory = str(
        sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), str(sys.argv[4])

    # Create the dataset and split into train and test:
    dataset = dataset(clipped_reefs_path, image_size)  # noqa
    X_train, X_test = dataset.return_data()

    print("Lambda is: ", lambda_)
    # Run the model
    run_model(X_train, X_test, image_size, lambda_, save_directory)
