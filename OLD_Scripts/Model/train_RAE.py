# File to train the Robust Autoencoder:
import sys
from Robust_autoencoder import *  # noqa
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Show if the GPU is present: 
from tensorflow.python.client import device_lib
print(tf.config.list_physical_devices('GPU'))

def load_data(path, size):
    # Load the data:
    data = []
    for i in range(951):
         # f"Reef_data_numpy/75_pix_reefs_PADDED/Reef_{i}.npy", allow_pickle=True)
        loaded_array = np.load(path + f"/Reef_{i}.npy", allow_pickle = True)
        data.append(loaded_array)

    data = np.array(data)
    # Change Nan Values to 0 or Black:
    print("Before Removing NaNs there are: ", sum(sum(sum(np.isnan(data)))))
    data[np.isnan(data) == True] = 0  # noqa
    print(f"Now there are {sum(sum(sum(np.isnan(data))))} NaNs")

    # Plot images:
    # plt.imshow(data[1])

    # reshape the arrays to include the padding
    data = data.reshape(951, (size)*(size))
    print(data.shape)
    print(data[0])
    return data


# Run the model:
def run_model(data, input_size, save_directory):
    # Check if GPU is available
    if tf.config.list_physical_devices('GPU'):
        # Specify GPU device
        with tf.device('/GPU:0'):
            with tf.compat.v1.Session() as sess:
                rae = RDAE(sess=sess, lambda_=4_000, layers_sizes=[input_size*input_size, 2_000, 1_500, 800])  # noqa
                L, S = rae.fit(data, sess=sess, learning_rate=.001, batch_size=40, inner_iteration=100,  # noqa
                               iteration=40, verbose=True)
                # Get the recognition from the input data
                recon_rae = rae.getRecon(data, sess=sess)
                # Save some examples:
                for i in range(20):
                    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                    ax[0].imshow(data[i].reshape(input_size, input_size), cmap='gray')
                    ax[0].set_title("Original Image")
                    ax[1].imshow(recon_rae[i].reshape(input_size, input_size), cmap='gray')
                    ax[1].set_title("Reconstructed Image")
                    plt.savefig((save_directory + f"example_{i}.png"))
                    plt.close(fig)
                
                # Reshape the L and S matrices: 
                L = L.reshape(951, input_size, input_size)
                S = S.reshape(951, input_size, input_size)
                for i in range(50):
                    # Plot the L and S matrices to see what they have captured
                    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                    ax[0].imshow(L[i], cmap='gray')
                    ax[0].set_title("Low Rank (L) Matrix")
                    ax[1].imshow(S[i], cmap='gray')
                    ax[1].set_title("Sparse (S) Matrix")
                    plt.savefig((save_directory + f"Final_Matrices{i}.png"))
                    plt.close(fig)


                # Save Model: Create a saver object
                saver = tf.compat.v1.train.Saver()
                saver.save(sess, save_directory)  # 'Model_Checkpoints/training_1/model')
                print("Model saved in path: %s" % save_directory)
    else:
        print("No GPU available - Please try again")

if __name__ == "__main__":
    # If statement to check that it is the right size: 
    if len(sys.argv) != 4:
        print("Usage: python script.py <Path/to/npy files> <SIZE of input image SIZExSIZE> <Path/to/Save Directory>")
        sys.exit(1)

    # Get variables from command line: 
    np_files_path, input_size, save_directory = str(sys.argv[1]), str(sys.argv[2]), str(sys.argv[3])

    # Load the data
    data = load_data(np_files_path, int(input_size))

    # Run the model
    run_model(data, int(input_size), save_directory)
