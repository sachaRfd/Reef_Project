# L21 Robust Auto-Encoder:
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def batches(data, n):
    """Yield successive n-sized batches from data, the last batch is the left indexes."""  # noqa
    for i in range(0, data, n):
        yield range(i, min(data, i+n))


class Deep_Autoencoder(object):
    def __init__(self, sess, input_dim_list=[784, 400]):
        """input_dim_list must include the original data dimension"""
        assert len(input_dim_list) >= 2
        self.W_list = []
        self.encoding_b_list = []
        self.decoding_b_list = []
        self.dim_list = input_dim_list
        self.learning_rate = tf.compat.v1.placeholder(tf.float32, shape=())

        # Encoders parameters
        for i in range(len(input_dim_list)-1):
            init_max_value = np.sqrt(
                6. / (self.dim_list[i] + self.dim_list[i+1]))
            self.W_list.append(tf.Variable(tf.random.uniform([self.dim_list[i], self.dim_list[i+1]],  # noqa
                                                             np.negative(init_max_value), init_max_value)))  # noqa
            self.encoding_b_list.append(tf.Variable(
                tf.random.uniform([self.dim_list[i+1]], -0.1, 0.1)))
        # Decoders parameters
        for i in range(len(input_dim_list)-2, -1, -1):
            self.decoding_b_list.append(tf.Variable(
                tf.random.uniform([self.dim_list[i]], -0.1, 0.1)))

        # Placeholder for input and dropout probability
        self.input_x = tf.compat.v1.placeholder(
            tf.float32, [None, self.dim_list[0]])

        # coding graph, including the new dropout layers:
        last_layer = self.input_x
        for weight, bias in zip(self.W_list, self.encoding_b_list):
            hidden = tf.sigmoid(tf.matmul(last_layer, weight) + bias)
            # hidden = tf.layers.dropout(hidden, 0.5)
            hidden = tf.keras.layers.Dropout(0.8)(hidden)
            last_layer = hidden
        self.hidden = hidden

        # decode graph, including the new dropout layer
        for weight, bias in zip(reversed(self.W_list), self.decoding_b_list):
            hidden = tf.sigmoid(
                tf.matmul(last_layer, tf.transpose(weight)) + bias)
            # hidden = tf.layers.dropout(hidden, 0.5)
            hidden = tf.keras.layers.Dropout(0.8)(hidden)
            last_layer = hidden
        self.recon = last_layer

        self.cost = 200 * tf.reduce_mean(tf.square(self.input_x - self.recon))
        self.cost_test = 200 * \
            tf.reduce_mean(tf.square(self.input_x - self.recon))
#         self.cost = 200*tf.losses.log_loss(self.recon, self.input_x)
        self.train_step = tf.compat.v1.train.AdamOptimizer(
            self.learning_rate).minimize(self.cost)
        sess.run(tf.compat.v1.global_variables_initializer())

    def fit(self, X, X_test, sess, learning_rate=0.15,
            iteration=200, batch_size=50, init=False, verbose=False):

        # Setup the list of costs:
        cost_list = []
        cost_list_test = []
        # Check that shape of the data matches the first layer of the model
        assert X.shape[1] == self.dim_list[0]
        # Check that the shape of the data matches too
        assert X_test.shape[1] == self.dim_list[0]
        if init:
            sess.run(tf.global_variables_initializer())
        sample_size = X.shape[0]
        for i in range(iteration):
            for one_batch in batches(sample_size, batch_size):
                sess.run(self.train_step, feed_dict={
                         self.input_x: X[one_batch], self.learning_rate: learning_rate})  # noqa
            if verbose and i % 10 == 0:
                e = self.cost.eval(session=sess, feed_dict={self.input_x: X})
                print("    iteration : ", i, ", cost : ", e)
                e_test = self.cost_test.eval(session=sess, feed_dict={
                                             self.input_x: X_test})
                print("    iteration : ", i, ", cost_test : ", e_test)
                print()
            # Append the cost to the list so we can plot the convergence plot:
            e = self.cost.eval(session=sess, feed_dict={self.input_x: X})
            cost_list.append(e)

            # Same for Test Set:
            e_test = self.cost_test.eval(
                session=sess, feed_dict={self.input_x: X_test})
            cost_list_test.append(e_test)

        return cost_list, cost_list_test

    def transform(self, X, sess):
        return self.hidden

    def getRecon(self, X, sess):
        return self.recon.eval(session=sess, feed_dict={self.input_x: X})


def l21shrink(epsilon, x):
    """
    auther : Chong Zhou
    date : 10/20/2016
    update to python3: 03/15/2019
    Args:
        epsilon: the shrinkage parameter
        x: matrix to shrink on
    Ref:
        wiki Regularization: {https://en.wikipedia.org/wiki/Regularization_(mathematics)}  # noqa
    Returns:
            The shrunk matrix
    """
    output = x.copy()
    norm = np.linalg.norm(x, ord=2, axis=0)
    for i in range(x.shape[1]):
        if norm[i] > epsilon:
            for j in range(x.shape[0]):
                output[j, i] = x[j, i] - epsilon * x[j, i] / norm[i]
        else:
            output[:, i] = 0.
    return output


class RobustL21Autoencoder(object):
    """
    @author: Chong Zhou
    first version.
    complete: 10/20/2016
    Updated to python3
    Des:
        X = L + S
        L is a non-linearly low dimension matrix and S is a sparse matrix.
        argmin ||L - Decoder(Encoder(L))|| + ||S||_2,1
        Use Alternating projection to train model
        The idea of shrink the l21 norm comes from the wiki 'Regularization' link: {  # noqa
            https://en.wikipedia.org/wiki/Regularization_(mathematics)
        }
    """

    def __init__(self, sess, layers_sizes, lambda_=1.0, error=1.0e-8):
        """
        sess: a Tensorflow tf.Session object
        layers_sizes: a list that contain the deep ae layer sizes, including the input layer  # noqa
        lambda_: tuning the weight of l1 penalty of S
        error: converge criterior for jump out training iteration
        """
        self.lambda_ = lambda_
        self.layers_sizes = layers_sizes
        self.error = error
        self.errors = []
        self.AE = Deep_Autoencoder(sess=sess, input_dim_list=self.layers_sizes)

    def fit(self, X, X_test, sess, save_directory, learning_rate=0.15, inner_iteration=50,  # noqa
            iteration=20, batch_size=133, re_init=False, verbose=False):
        # The first layer must be the input layer, so they should have same sizes.  # noqa
        assert X.shape[1] == self.layers_sizes[0]
        # initialize L, S
        self.L = np.zeros(X.shape)
        self.S = np.zeros(X.shape)

        if verbose:
            print("X shape: ", X.shape)
            print("L shape: ", self.L.shape)
            print("S shape: ", self.S.shape)

        # Setup list of costs for plotting:
        list_of_costs = []
        list_of_costs_test = []
        for it in range(iteration):
            if verbose:
                print("Out iteration: ", it)
            # alternating project, first project to L
            self.L = X - self.S
            # Using L to train the auto-encoder
            costs, costs_test = self.AE.fit(X=self.L, X_test=X_test, sess=sess,
                                            iteration=inner_iteration,
                                            learning_rate=learning_rate,
                                            batch_size=batch_size,
                                            init=re_init,
                                            verbose=verbose)
            # Append the Costs to list for visualisation:
            list_of_costs.append(costs)
            list_of_costs_test.append(costs_test)

            # get optmized L
            self.L = self.AE.getRecon(X=self.L, sess=sess)
            # alternating project, now project to S and shrink S
            self.S = l21shrink(self.lambda_, (X - self.L).T).T

            # Show examples of True, Reconstruction, L and Sparse Matrix:
            # Get recon  # Before was X_test but we will keep it
            recon_ = self.getRecon(X, sess)

            # Save Plots every 20 iterations:
            if it % 20 == 0:
                directory_name = save_directory + \
                    f'Examples_{it}_iteration'
                if not os.path.exists(directory_name):
                    os.mkdir(directory_name)
                # Plot true images:
                fig, ax = plt.subplots(10, 10, figsize=(20, 20))
                for i in range(100):
                    row, col = divmod(i, 10)
                    ax[row][col].imshow(np.reshape(X[i, :], (28, 28)))
                    ax[row][col].axis('off')
                plt.suptitle("True Images")
                plt.savefig(f"{directory_name}/true_images.png")
                plt.close(fig)

                # Plot reconstructed images
                fig, ax = plt.subplots(10, 10, figsize=(20, 20))
                for i in range(100):
                    row, col = divmod(i, 10)
                    ax[row][col].imshow(np.reshape(recon_[i, :], (28, 28)))
                    ax[row][col].axis('off')
                plt.suptitle("Reconstructed Images")
                plt.savefig(f"{directory_name}/reconstructed_images.png")
                plt.close(fig)

                # Plot L matrices
                fig, ax = plt.subplots(10, 10, figsize=(20, 20))
                for i in range(100):
                    row, col = divmod(i, 10)
                    ax[row][col].imshow(np.reshape(self.L[i, :], (28, 28)))
                    ax[row][col].axis('off')
                plt.suptitle("L Matrices")
                plt.savefig(f"{directory_name}/L_matrices.png")
                plt.close(fig)

                # Plot S matrices
                fig, ax = plt.subplots(10, 10, figsize=(20, 20))
                for i in range(100):
                    row, col = divmod(i, 10)
                    ax[row][col].imshow(np.reshape(self.S[i, :], (28, 28)))
                    ax[row][col].axis('off')
                plt.suptitle("S Matrices")
                plt.savefig(f"{directory_name}/S_matrices.png")
                plt.close(fig)

        return self.L, self.S, list_of_costs, list_of_costs_test

    def transform(self, X, sess):
        L = X - self.S
        return self.AE.transform(X=L, sess=sess)

    def getRecon(self, X, sess):
        return self.AE.getRecon(X, sess=sess)
