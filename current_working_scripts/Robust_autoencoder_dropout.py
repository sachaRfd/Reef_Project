# Robust AutoEncoder Model for reef data:
import os
import tensorflow as tf
import numpy.linalg as nplin
import numpy as np
import matplotlib.pyplot as plt
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()


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


def shrink(epsilon, x):
    """
    @Original Author: Prof. Randy
    @Modified by: Chong Zhou
    update to python3: 03/15/2019
    Args:
        epsilon: the shrinkage parameter (either a scalar or a vector)
        x: the vector to shrink on

    Returns:
        The shrunk vector
    """
    output = np.array(x*0.)

    for idx, ele in enumerate(x):
        if ele > epsilon:
            output[idx] = ele - epsilon
        elif ele < -epsilon:
            output[idx] = ele + epsilon
        else:
            output[idx] = 0.
    return output


class RDAE(object):
    """
    @author: Chong Zhou
    2.0 version.
    complete: 10/17/2016
    version changes: move implementation from theano to tensorflow.
    3.0
    complete: 2/12/2018
    changes: delete unused parameter, move shrink function to other file
    update: 03/15/2019
        update to python3
    Des:
        X = L + S
        L is a non-linearly low rank matrix and S is a sparse matrix.
        argmin ||L - Decoder(Encoder(L))|| + ||S||_1
        Use Alternating projection to train model
    """

    def __init__(self, sess, layers_sizes, lambda_=1.0, error=1.0e-7):
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
        self.AE = Deep_Autoencoder(
            sess=sess, input_dim_list=self.layers_sizes)

    def fit(self, X, X_test, sess, save_directory, learning_rate=0.15, inner_iteration=50,
            iteration=20, batch_size=50, verbose=False):
        # The first layer must be the input layer, so they should have same sizes.  # noqa
        assert X.shape[1] == self.layers_sizes[0]

        # initialize L, S, mu(shrinkage operator)
        self.L = np.zeros(X.shape)
        self.S = np.zeros(X.shape)

        # Add this to avoid division by 0q
        mu = (X.size) / ((4.0 * nplin.norm(X, 1)))
        print("Mu is", mu)
        print("shrink parameter:", self.lambda_ / mu)
        LS0 = self.L + self.S

        XFnorm = nplin.norm(X, 'fro')
        if verbose:
            print("X shape: ", X.shape)
            print("L shape: ", self.L.shape)
            print("S shape: ", self.S.shape)
            print("mu: ", mu)
            print("XFnorm: ", XFnorm)

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
                                            verbose=verbose)

            list_of_costs.append(costs)
            list_of_costs_test.append(costs_test)
            # get optmized L
            self.L = self.AE.getRecon(X=self.L, sess=sess)
            # alternating project, now project to S
            self.S = shrink(
                self.lambda_/mu, (X - self.L).reshape(X.size)).reshape(X.shape)

            # break criterion 1: the L and S are close enough to X
            c1 = nplin.norm(X - self.L - self.S, 'fro') / XFnorm
            # break criterion 2: there is no changes for L and S
            c2 = np.min([mu, np.sqrt(mu)]) * \
                nplin.norm(LS0 - self.L - self.S) / XFnorm

            if verbose:
                print("c1: ", c1)
                print("c2: ", c2)

            if c1 < self.error and c2 < self.error:
                print("early break")
                break
            # save L + S for c2 check in the next iteration
            LS0 = self.L + self.S

            # Get examples after each outer iteration
            for i in range(0, 100, 10): 
                if i % 2 == 0:
                    # Create a directory to store the reconstructions:
                    directory_name = save_directory + f'reconstruction_examples_{it}_iteration'
                    if not os.path.exists(directory_name):
                        os.mkdir(directory_name)


                    recon = self.getRecon(X_test, sess)
                    fig, ax = plt.subplots(2, 10, figsize=(10, 2))
                    for example_i in range(10):
                        ax[0][example_i].imshow(
                            np.reshape(X_test[example_i + i, :], (28, 28)))  # change this 
                        ax[1][example_i].imshow(
                            np.reshape(recon[example_i + i, :], (28, 28)))  
                    plt.savefig('{}/{}.png'.format(directory_name, str(i) + "_example_batch"), bbox_inches='tight')  
                    plt.close(fig)

                    # Create a Directory to store the sparse matrices: 
                    directory_name_sparse = save_directory + f'reconstruction_sparse_matrix_{it}_iteration'
                    if not os.path.exists(directory_name_sparse):
                        os.mkdir(directory_name_sparse)
                    fig, ax = plt.subplots(2, 10, figsize=(10, 2))
                    for example_i in range(10):
                        ax[0][example_i].imshow(
                            np.reshape(self.L[example_i + i, :], (28, 28)))
                        ax[1][example_i].imshow(
                            np.reshape(self.S[example_i + i, :], (28, 28)))  
                    plt.savefig('{}/{}.png'.format(directory_name_sparse, str(i) + "_L_and_S_matrices"), bbox_inches='tight')  
                    plt.close(fig)

        return self.L, self.S, list_of_costs, list_of_costs_test

    def transform(self, X, sess):
        L = X - self.S
        return self.AE.transform(X=L, sess=sess)

    def getRecon(self, X, sess):
        return self.AE.getRecon(X, sess=sess)
