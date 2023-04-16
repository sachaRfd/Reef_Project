from Robust_autoencoder import *  # noqa
import sys
# import tensorflow as tf
import tensorflow.compat.v1 as tf
import numpy as np
import rasterio

X = np.random.rand(1, 75*75)

def generate_output(save_path, input_size, output_path):  # Model_Checkpoints/training_1/ 75 Model_Output/test_1/  # noqa

    # create a new session to restore the saved model
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(save_path + "model.meta")  # noqa
        saver.restore(sess, tf.train.latest_checkpoint(save_path))  # noqa
        print("Model Restored")
        ae = Deep_Autoencoder(sess=sess, input_dim_list=[input_size*input_size, 100])  # noqa

        # Get the reconstructed output for your desired input data and save it in the output folder Model_Output/test_1  # noqa
        reconstructed_output = ae.getRecon(X, sess)
        np.save(output_path + "reconstructed_output.npy", reconstructed_output)  # noqa
        # Plot the reconstructed output in the folder Model_Output/test_1 as tiff file:  # noqa
        reconstructed_output = reconstructed_output.reshape(75, 75)
        with rasterio.open(output_path + "reconstructed_output.tiff", "w", driver="GTiff",  # noqa
                              width=75, height=75, count=1, dtype=reconstructed_output.dtype) as dst:  # noqa
            dst.write(reconstructed_output, 1)
        print(f"Reconstructed output saved in the folder {output_path}")

        # Plot the output:
        # plt.imshow(reconstructed_output.reshape(75, 75))

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <Path/to/npy files> <SIZE of input image SIZExSIZE> <Path/to/Save Directory>")
        sys.exit(1)
    
    # Get variables from command line: 
    save_path, input_size, output_directory = str(sys.argv[1]), str(sys.argv[2]), str(sys.argv[3])

    generate_output(save_path, int(input_size), output_directory)
