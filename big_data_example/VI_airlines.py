###############################################################################
################## VARIATIONAL INFERENCE ALGORITHM WITH NNs ###################
###############################################################################
#
# This code performs VI in the Airlines Delay dataset, testing the convergence
# of the algorithm during its training period
#
###############################################################################
###############################################################################


# Import the relevant packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import time
from datetime import datetime

import tensorflow as tf
import numpy as np

import os
os.chdir(".")


# =============================================================================
# Complete system parameters (VAE, encoder, decoder, data specs, etc.)
# =============================================================================

# File to be analyzed
original_file = "shuffle_airlines.npy"

# This is the total number of training samples
total_training_data = 1.0
n_batch = 100
n_epochs = 150 # We do not expect the algorithm to fulfill more than a few epochs before it reaches convergence
n_data_test = 10000
sampling_batches = 100       # How often do we want to sample the batches for results in the RMSE and LL

mean_targets = 0.0
std_targets = 1.0

# Number of samples for the weights in each layer
samples_train = 10
samples_test = 100

# Structural parameters of the main NN
dim_data = 8
n_units = 50
n_units_sec = 50
total_number_weights = (dim_data + 1) * n_units  # Total number of weights used
total_number_weights_double = n_units * (dim_data + n_units_sec) + n_units_sec

# Learning ratios
primal_rate = 1e-5
dual_rate = 1e-4

# Create the model
x = tf.placeholder(tf.float32, [ None, dim_data ])

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [ None, 1 ])

# We define the following two functions to simplify the rest of the code
def w_variable_mean(shape):
  initial = tf.random_normal(shape = shape, stddev = 0.1) # mean 0 stddev 1
  return tf.Variable(initial)

def w_variable_variance(shape):
  initial = tf.random_normal(shape = shape, stddev = 0.1) - 5.0 # mean 0 stddev 1
  return tf.Variable(initial)


###############################################################################
########################## Estructura de la red ###############################
###############################################################################

# =============================================================================
#  (Encoder) Deterministic NN that generates the weights
# =============================================================================

# Case with one layer

def generate_weights(batchsize, samples_train, samples_test):

    # Inputs
    # batchsize :   Dimension of the batch of data we are going to apply the weights to
    # samples   :   Number of samples of the weights required
    #

    # Initialize the random gaussian noises to input the network


    init_noise_train  = tf.random_normal(shape = [ batchsize, samples_train, (dim_data + 1) * n_units ])
    init_noise_test = tf.random_normal(shape = [ batchsize, samples_test, (dim_data + 1) * n_units ])

    means = w_variable_mean([ 1, 1, (dim_data + 1) * n_units ])
    log_variances = w_variable_variance([ 1, 1, (dim_data + 1) * n_units ])

    result_train = means + tf.sqrt(tf.exp(log_variances)) * init_noise_train
    result_test = means + tf.sqrt(tf.exp(log_variances)) * init_noise_test

    return result_train, result_test, [ means, log_variances ]


# Case with two layers

def generate_weights_double(batchsize, samples_train, samples_test):

    # Inputs
    # batchsize :   Dimension of the batch of data we are going to apply the weights to
    # samples   :   Number of samples of the weights required
    #

    # Initialize the random gaussian noises to input the network


    init_noise_train  = tf.random_normal(shape = [ batchsize, samples_train, total_number_weights_double ])
    init_noise_test = tf.random_normal(shape = [ batchsize, samples_test, total_number_weights_double ])

    means = w_variable_mean([ 1, 1, total_number_weights_double ])
    log_variances = w_variable_variance([ 1, 1, total_number_weights_double ])

    result_train = means + tf.sqrt(tf.exp(log_variances)) * init_noise_train
    result_test = means + tf.sqrt(tf.exp(log_variances)) * init_noise_test

    return result_train, result_test, [ means, log_variances ]

# =============================================================================
# (VAE) Main network
# =============================================================================

# Case with 1 layer

def exp_log_likelihood(mean_targets, std_targets, weights_train, weights_test):

    # This is the noise variance

    log_sigma2_noise = tf.Variable(tf.cast(1.0 / 100.0, dtype = tf.float32))

    # Separate the weights and reshape the tensors

#    W1_train = tf.reshape(weights_train[:,:,:(dim_data * n_units)], shape = [ tf.shape(x)[0], samples_train, n_units, dim_data ])
#    W2_train  = tf.reshape(weights_train[:,:,(dim_data * n_units):], shape = [ tf.shape(x)[0], samples_train, n_units, 1 ])

    W1_train_re = tf.reshape(weights_train[:,:,:(dim_data * n_units)], shape = [ tf.shape(x)[0], samples_train, n_units, dim_data ])
    W2_train_re  = tf.reshape(weights_train[:,:,(dim_data * n_units):], shape = [ tf.shape(x)[0], samples_train, n_units, 1 ])

    W1_test_re = tf.reshape(weights_test[:,:,:(dim_data * n_units)], shape = [ tf.shape(x)[0], samples_test, n_units, dim_data ])
    W2_test_re  = tf.reshape(weights_test[:,:,(dim_data * n_units):], shape = [ tf.shape(x)[0], samples_test, n_units, 1 ])

    #########################
    #   Processing layer    #
    #########################

    bias_A1 = w_variable_mean([ n_units ])

    A1 = tf.reduce_sum(tf.reshape(x, shape = [tf.shape(x)[0], 1, 1, dim_data]) * W1_train_re, axis = 3) + bias_A1
    A1_test = tf.reduce_sum(tf.reshape(x, shape = [tf.shape(x)[0], 1, 1, dim_data]) * W1_test_re, axis = 3) + bias_A1

    # Results of the layer

    h1 = tf.nn.leaky_relu(A1)
    h1_test = tf.nn.leaky_relu(A1_test)

    ##############################
    #    Regression output       #
    ##############################

    bias_A2  = w_variable_mean([ 1 ])

    # Results of the layer

    A2 = tf.reduce_sum(tf.reshape(h1, shape = [ tf.shape(x)[0], samples_train, n_units, 1 ]) * W2_train_re, axis = 2) + bias_A2
    A2_test = tf.reduce_sum(tf.reshape(h1_test, shape = [ tf.shape(x)[0], samples_test, n_units, 1 ]) * W2_test_re, axis = 2) + bias_A2

    res_train = tf.reduce_mean(-0.5 * (np.log(2.0 * np.pi) + log_sigma2_noise + (A2 - tf.reshape(y_, shape = [ tf.shape(x)[0], 1, 1 ]))**2 / tf.exp(log_sigma2_noise)), axis = [ 1 ])

    # This is to compute the test log loglikelihood

    log_prob_data_test = tf.reduce_sum(((tf.reduce_logsumexp(-0.5 * tf.log(2.0 * np.pi * (tf.exp(log_sigma2_noise) * std_targets**2)) \
        - 0.5 * (A2_test * std_targets + mean_targets - tf.reshape(y_, shape = [ tf.shape(x)[0], 1, 1 ]))**2 / (tf.exp(log_sigma2_noise) * std_targets**2), axis = [ 1 ])) \
        - np.log(samples_test)))

    squared_error = tf.reduce_sum((tf.reduce_mean(A2_test, axis = [ 1 ]) * std_targets + mean_targets - y_ )**2)

    return res_train, [ log_sigma2_noise, bias_A1, bias_A2 ], squared_error, log_prob_data_test


# Case with 2 layers

def exp_log_likelihood_double(mean_targets, std_targets, weights_train, weights_test):

    # This is the noise variance

    log_sigma2_noise = tf.Variable(tf.cast(1.0 / 100.0, dtype = tf.float32))

    # Separate the weights and reshape the tensors

    W1_train_re = tf.reshape(weights_train[:,:,:(dim_data * n_units)], shape = [ tf.shape(x)[0], samples_train, n_units, dim_data ])
    W2_train_re  = tf.reshape(weights_train[:,:,(dim_data * n_units):(n_units * (n_units_sec + dim_data))], shape = [ tf.shape(x)[0], samples_train, n_units, n_units_sec ])
    W3_train_re  = tf.reshape(weights_train[:,:,(n_units * (n_units_sec + dim_data)):], shape = [ tf.shape(x)[0], samples_train, n_units_sec, 1 ])

    W1_test_re = tf.reshape(weights_test[:,:,:(dim_data * n_units)], shape = [ tf.shape(x)[0], samples_test, n_units, dim_data ])
    W2_test_re  = tf.reshape(weights_test[:,:,(dim_data * n_units):(n_units * (n_units_sec + dim_data))], shape = [ tf.shape(x)[0], samples_test, n_units, n_units_sec ])
    W3_test_re  = tf.reshape(weights_test[:,:,(n_units * (n_units_sec + dim_data)):], shape = [ tf.shape(x)[0], samples_test, n_units_sec, 1 ])

    #############################
    # First layer of processing #
    #############################

    bias_A1 = w_variable_mean([ n_units ])

    A1 = tf.reduce_sum(tf.reshape(x, shape = [tf.shape(x)[0], 1, 1, dim_data]) * W1_train_re, axis = 3) + bias_A1
    A1_test = tf.reduce_sum(tf.reshape(x, shape = [tf.shape(x)[0], 1, 1, dim_data]) * W1_test_re, axis = 3) + bias_A1

    # Results of the layer

    h1 = tf.nn.leaky_relu(A1)
    h1_test = tf.nn.leaky_relu(A1_test)

    #######################
    # Second hidden layer #
    #######################

    bias_A2  = w_variable_mean([ n_units_sec ])

    A2 = tf.reduce_sum(tf.reshape(h1, shape = [ tf.shape(x)[0], samples_train, n_units, 1 ]) * W2_train_re, axis = 2) + bias_A2
    A2_test = tf.reduce_sum(tf.reshape(h1_test, shape = [ tf.shape(x)[0], samples_test, n_units, 1 ]) * W2_test_re, axis = 2) + bias_A2

    # Results of the layer

    h2 = tf.nn.leaky_relu(A2)
    h2_test = tf.nn.leaky_relu(A2_test)

    ################
    # Output layer #
    ################

    bias_A3  = w_variable_mean([ 1 ])

    A3 = tf.reduce_sum(tf.reshape(h2, shape = [ tf.shape(x)[0], samples_train, n_units_sec, 1 ]) * W3_train_re, axis = 2) + bias_A3
    A3_test = tf.reduce_sum(tf.reshape(h2_test, shape = [ tf.shape(x)[0], samples_test, n_units_sec, 1 ]) * W3_test_re, axis = 2) + bias_A3

    res_train = tf.reduce_mean(-0.5 * (np.log(2.0 * np.pi) + log_sigma2_noise + (A3 - tf.reshape(y_, shape = [ tf.shape(x)[0], 1, 1 ]))**2 / tf.exp(log_sigma2_noise)), axis = [ 1 ])

    # This is to compute the test log loglikelihood

    log_prob_data_test = tf.reduce_sum(((tf.reduce_logsumexp(-0.5 * tf.log(2.0 * np.pi * (tf.exp(log_sigma2_noise) * std_targets**2)) \
        - 0.5 * (A3_test * std_targets + mean_targets - tf.reshape(y_, shape = [ tf.shape(x)[0], 1, 1 ]))**2 / (tf.exp(log_sigma2_noise) * std_targets**2), axis = [ 1 ])) \
        - np.log(samples_test)))

    squared_error = tf.reduce_sum((tf.reduce_mean(A3_test, axis = [ 1 ]) * std_targets + mean_targets - y_ )**2)


    return res_train, [ log_sigma2_noise, bias_A1, bias_A2, bias_A3 ], squared_error, log_prob_data_test


###############################################################################
###############################################################################
###############################################################################


def main(layers):

    np.random.seed(123)
    tf.set_random_seed(123)

    # We load the original dataset
    data = np.load(original_file)

    # =========================================================================
    #  Parameters of the complete system
    # =========================================================================

    # We obtain the features and the targets

    X = data[ :, range(data.shape[ 1 ] - 1) ]
    y = data[ :, data.shape[ 1 ] - 1 ]

    # We create the train and test sets with 90% and 10% of the data

    data_size = X.shape[ 0 ]
    size_train = data_size - n_data_test
    total_training_data = size_train

    X_train = X[  0 : size_train, : ]
    y_train = np.vstack(y[  0 : size_train ])
    X_test = X[ size_train : data_size, : ]
    y_test = np.vstack(y[ size_train : data_size ])

    # Normalize the arguments

    meanXTrain = np.mean(X_train, axis = 0)
    stdXTrain = np.std(X_train, axis = 0)

    meanyTrain = np.mean(y_train)
    stdyTrain = np.std(y_train)

    X_train = (X_train - meanXTrain) / stdXTrain
    y_train = (y_train - meanyTrain) / stdyTrain
    X_test = (X_test - meanXTrain) / stdXTrain

    std_targets = stdyTrain
    mean_targets = meanyTrain

    # =========================================================================
    #  Calculations in the NNs
    # =========================================================================

    # Generate the weights
    if layers == 1:
        weights_train, weights_test, vars_gen = generate_weights( tf.shape(x)[ 0 ], samples_train, samples_test)
    elif layers == 2:
        weights_train, weights_test, vars_gen = generate_weights_double( tf.shape(x)[ 0 ], samples_train, samples_test)

    # Calculate the error metrics in the VAE
    if layers == 1:
        res_train, vars_network, squared_error, log_prob_data_test = exp_log_likelihood(mean_targets, std_targets, weights_train, weights_test)
    elif layers == 2:
        res_train, vars_network, squared_error, log_prob_data_test = exp_log_likelihood_double(mean_targets, std_targets, weights_train, weights_test)

    log_vars_prior = w_variable_variance([ 1 ])

    KL =  0.5 * tf.reduce_sum(tf.exp(vars_gen[ 1 ] - log_vars_prior) + (0.0 - vars_gen[ 0 ])**2 / tf.exp(log_vars_prior) - 1.0 + log_vars_prior - vars_gen[ 1 ])

    ELBO = tf.reduce_sum(res_train) - KL * tf.cast(tf.shape(x)[ 0 ], tf.float32) / tf.cast(total_training_data, tf.float32)
    neg_ELBO = -ELBO
    main_loss = neg_ELBO
    mean_ELBO = neg_ELBO

    # KL y res_train have shape batch_size x n_samples

    mean_KL = KL

    # Create the variable lists to be updated

    vars_primal = vars_gen + vars_network + [ log_vars_prior ]

    train_step_primal = tf.train.AdamOptimizer(primal_rate).minimize(main_loss, var_list = vars_primal)

    # Calculate the squared error

    timing = list()
    tmp_kl = list()
    tmp_elbo = list()

    config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1, allow_soft_placement=True, device_count = {'CPU': 1})

    with tf.Session(config = config) as sess:

        sess.run(tf.global_variables_initializer())
        total_ini = time.time()
        for epoch in range(n_epochs):

            for i_batch in range(int(np.ceil(size_train / n_batch))):

                L = 0.0

                ini = time.clock()
                ini_ref = time.time()
                ini_train = time.clock()

                last_point = np.minimum(n_batch * (i_batch + 1), X_train.shape[ 0 ])

                batch = [ X_train[ i_batch * n_batch : last_point, : ] , y_train[ i_batch * n_batch : last_point, ] ]

#                sess.run(train_step_dual, feed_dict={x: batch[0], y_: batch[1]})
                sess.run(train_step_primal, feed_dict={x: batch[0], y_: batch[1]})

                value = sess.run(mean_ELBO, feed_dict={x: batch[0], y_: batch[1]})
                kl = sess.run(mean_KL, feed_dict={x: batch[0], y_: batch[1]})
                L += value

                fini_train = time.clock()

                if i_batch % sampling_batches == 0:

                    ini_test = time.time()

                    SE = 0.0
                    for i in range(int(np.ceil(n_data_test / n_batch))):

                        last_point = np.minimum(n_batch * (i + 1), n_data_test)

                        batch = [ X_test[ i * n_batch : last_point, : ] , y_test[ i * n_batch : last_point, ] ]

                        SE += sess.run(squared_error, feed_dict={x: batch[0], y_: batch[1]}) / batch[ 0 ].shape[ 0 ]

                    RMSE = np.sqrt(SE / int(np.ceil(n_data_test / n_batch)))

                    # We do the test evaluation RMSE

                    LL = 0.0
                    for j in range(int(np.ceil(n_data_test / n_batch))):

                        last_point = np.minimum(n_batch * (j + 1), n_data_test)

                        batch = [ X_test[ j * n_batch : last_point, : ] , y_test[ j * n_batch : last_point, ] ]

                        LL += sess.run(log_prob_data_test, feed_dict={x: batch[0], y_: batch[1]}) / batch[ 0 ].shape[ 0 ]

                    TestLL = (LL / int(np.ceil(n_data_test / n_batch)))

                    fini = time.clock()
                    fini_ref = time.time()
                    fini_test = time.time()
                    total_fini = time.time()

                    with open("results_VI_airlines/res_vi.txt", "a") as res_file:
                        res_file.write('VI batch %g datetime %s epoch %d ELBO %g KL %g real_time %g cpu_time %g train_time %g test_time %g total_time %g LL %g RMSE %g' % (i_batch, str(datetime.now()), epoch, L, kl, (fini_ref - ini_ref), (fini - ini), (fini_train - ini_train), (fini_test - ini_test), (total_fini - total_ini), TestLL, RMSE) + "\n")


#            timing.append(fini-ini)
#            tmp_kl.append(kl)
#            tmp_elbo.append(L)



if __name__ == '__main__':

    if not os.path.isdir("results_VI_airlines"):
        os.makedirs("results_VI_airlines")

    if os.path.isfile("results_VI_airlines/res_vi.txt"):
        with open("results_VI_airlines/res_vi.txt", "w") as file:
            file.close()

    layers = int(sys.argv[ 1 ])

    # Call main (in this case, no splits are needed and therefore there are no arguments)
    main(layers)
