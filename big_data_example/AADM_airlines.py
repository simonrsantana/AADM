###############################################################################
############## ADVERSARIAL ALPHA DIVERGENCE MINIMIZATION WITH NN ##############
###############################################################################
#
# This code performs AADM in the Airlines Delay dataset, testing the convergence
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
n_epochs = 1000 # We do not expect the algorithm to fulfill more than a few epochs before it reaches convergence
n_size_test = 10000
sampling_batches = 1000

mean_targets = 0.0
std_targets = 1.0

# Number of samples for the weights in each layer
samples_train = 10
samples_test = 100

# Structural parameters of the main NN
dim_data = 8
n_units = 50
n_units_sec = 50    # For the case with 2 hidden layers
total_number_weights = (dim_data + 1) * n_units  # Total number of weights used
total_number_weights_double = n_units * (dim_data + n_units_sec) + n_units_sec  # Number of weights for the 2 hidden layers case

# Parameters for the generator network
n_units_gen = 50          # Number of units in the generator for 1 hidden layer in the VAE
n_units_gen_2 = 50
n_units_gen_double = 50   # Number of units in the generator for 2 hidden layers in the VAE
n_units_gen_double_2 = 50
noise_comps_gen = 100     # Number of gaussian variables inputed in the encoder

# Parameters for the discriminative network
n_units_disc = 50
n_units_disc_2 = 50

# Learning ratios
primal_rate = 1e-4
dual_rate = 1e-3

# Create the model
x = tf.placeholder(tf.float32, [ None, dim_data ])

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [ None, 1 ])

# We define the following two functions to simplify the rest of the code
def w_variable_mean(shape):
  initial = tf.random_normal(shape = shape, stddev = 0.1, seed = 123) # mean 0 stddev 1
  return tf.Variable(initial)

def w_variable_variance(shape):
  initial = tf.random_normal(shape = shape, stddev = 0.1, seed = 123) - 5.0 # mean 0 stddev 1
  return tf.Variable(initial)


###############################################################################
##########################   Network structure  ###############################
##############################################################################a

##############################
# Case with one hidden layer #
##############################


def generate_weights(batchsize, samples_train, samples_test):

    # Inputs
    # batchsize :   Dimension of the batch of data we are going to apply the weights to
    # samples   :   Number of samples of the weights required
    #

    # Initialize the random gaussian noises to input the network

    pre_init_noise_train  = tf.random_normal(shape = [ batchsize, samples_train, noise_comps_gen ], seed = 123)
    pre_init_noise_test = tf.random_normal(shape = [ batchsize, samples_test, noise_comps_gen ], seed = 123)

    mean_noise = w_variable_mean([ 1, 1, noise_comps_gen ])
    log_var_noise = w_variable_variance([ 1, 1, noise_comps_gen ])

    init_noise_train =  mean_noise + tf.sqrt(tf.exp( log_var_noise )) * pre_init_noise_train
    init_noise_test = mean_noise + tf.sqrt(tf.exp( log_var_noise )) * pre_init_noise_test

    #Process the noises through the network
    W1_gen = w_variable_mean([ noise_comps_gen, n_units_gen ])
    bias1_gen  = w_variable_mean([ n_units_gen ])

    A1_gen_train = tf.tensordot(init_noise_train, W1_gen, axes = [[2], [0]]) + bias1_gen
    A1_gen_test = tf.tensordot(init_noise_test, W1_gen, axes = [[2], [0]]) + bias1_gen

    # Results for the first layer
    h1_gen_train = tf.nn.leaky_relu(A1_gen_train)
    h1_gen_test = tf.nn.leaky_relu(A1_gen_test)

    # Variables for the inner layer
    W2_gen = w_variable_mean([ n_units_gen, n_units_gen_2 ])
    bias2_gen  = w_variable_mean([ n_units_gen_2 ])

    A2_gen_train = tf.tensordot(h1_gen_train, W2_gen, axes = [[2], [0]]) + bias2_gen
    A2_gen_test = tf.tensordot(h1_gen_test, W2_gen, axes = [[2], [0]]) + bias2_gen

    # Results for the second layer
    h2_gen_train = tf.nn.leaky_relu(A2_gen_train)
    h2_gen_test = tf.nn.leaky_relu(A2_gen_test)

    # Output the weights
    W3_gen = w_variable_mean([ n_units_gen_2, total_number_weights])
    bias3_gen  = w_variable_mean([ total_number_weights ])

    A3_gen_train = tf.tensordot(h2_gen_train, W3_gen, axes = [[2], [0]]) + bias3_gen    # final weights
    A3_gen_test = tf.tensordot(h2_gen_test, W3_gen, axes = [[2], [0]]) + bias3_gen    # final weights

    return A3_gen_train, A3_gen_test, [ W1_gen, bias1_gen, W2_gen, bias2_gen, W3_gen, bias3_gen, mean_noise, log_var_noise] #, init_noise_train, init_noise_test ]


###############################
# Case with two hidden layers #
###############################


def generate_weights_double(batchsize, samples_train, samples_test):

    # Inputs
    # batchsize :   Dimension of the batch of data we are going to apply the weights to
    # samples   :   Number of samples of the weights required
    #

    # Initialize the random gaussian noises to input the network
    pre_init_noise_train  = tf.random_normal(shape = [ batchsize, samples_train, noise_comps_gen ], seed = 123)
    pre_init_noise_test = tf.random_normal(shape = [ batchsize, samples_test, noise_comps_gen ], seed = 123)

    mean_noise = w_variable_mean([ 1, 1, noise_comps_gen ])
    log_var_noise = w_variable_variance([ 1, 1, noise_comps_gen ])

    init_noise_train =  mean_noise + tf.sqrt(tf.exp( log_var_noise )) * pre_init_noise_train
    init_noise_test = mean_noise + tf.sqrt(tf.exp( log_var_noise )) * pre_init_noise_test

    # Process the noises through the network
    W1_gen = w_variable_mean([ noise_comps_gen, n_units_gen_double ])
    bias1_gen  = w_variable_mean([ n_units_gen_double ])

    A1_gen_train = tf.tensordot(init_noise_train, W1_gen, axes = [[2], [0]]) + bias1_gen
    A1_gen_test = tf.tensordot(init_noise_test, W1_gen, axes = [[2], [0]]) + bias1_gen

    # Results for the layer
    h1_gen_train = tf.nn.leaky_relu(A1_gen_train)
    h1_gen_test = tf.nn.leaky_relu(A1_gen_test)

    # Output the weights
    W2_gen = w_variable_mean([ n_units_gen_double, n_units_gen_double_2 ])
    bias2_gen  = w_variable_mean([ n_units_gen_double_2 ])

    A2_gen_train = tf.tensordot(h1_gen_train, W2_gen, axes = [[2], [0]]) + bias2_gen
    A2_gen_test = tf.tensordot(h1_gen_test, W2_gen, axes = [[2], [0]]) + bias2_gen

    # Results for the second layer
    h2_gen_train = tf.nn.leaky_relu(A2_gen_train)
    h2_gen_test = tf.nn.leaky_relu(A2_gen_test)

    # Output the weights
    W3_gen = w_variable_mean([ n_units_gen_double_2, total_number_weights_double ])
    bias3_gen  = w_variable_mean([ total_number_weights_double ])

    A3_gen_train = tf.tensordot(h2_gen_train, W3_gen, axes = [[2], [0]]) + bias3_gen    # final weights
    A3_gen_test = tf.tensordot(h2_gen_test, W3_gen, axes = [[2], [0]]) + bias3_gen    # final weights

    return A3_gen_train, A3_gen_test, [ W1_gen, bias1_gen, W2_gen, bias2_gen, W3_gen, bias3_gen, mean_noise, log_var_noise] #, init_noise_train, init_noise_test ]


# =============================================================================
#  (Decoder) Deterministic NN to output the value of T(z,x)
# =============================================================================

def discriminator(norm_weights_train, w_sampled_gaussian, layers):

    # Inputs
    # weights :     Tensor of rank 4 containing weight samples

    # Input the weights and process them
    if layers == 1:
        W1_disc = w_variable_mean([ total_number_weights, n_units_disc ])
    elif layers == 2:
        W1_disc = w_variable_mean([ total_number_weights_double, n_units_disc ])

    bias1_disc  = w_variable_mean([ n_units_disc ])

    A1_disc_norm_train = tf.tensordot(norm_weights_train, W1_disc, axes = [[2], [0]]) + bias1_disc
    A1_disc_gaussian = tf.tensordot(w_sampled_gaussian, W1_disc, axes = [[2], [0]]) + bias1_disc

    # Results for the first layer

    h1_disc_norm_train = tf.nn.leaky_relu(A1_disc_norm_train)
    h1_disc_gaussian = tf.nn.leaky_relu(A1_disc_gaussian)

    # Create the variables for the inner layer

    W2_disc = w_variable_mean([ n_units_disc, n_units_disc_2 ])
    bias2_disc  = w_variable_mean([ n_units_disc_2 ])

    A2_disc_norm_train = tf.tensordot(h1_disc_norm_train, W2_disc, axes = [[2], [0]]) + bias2_disc
    A2_disc_gaussian = tf.tensordot(h1_disc_gaussian, W2_disc, axes = [[2], [0]]) + bias2_disc

    # Results for the inner layer

    h2_disc_norm_train = tf.nn.leaky_relu(A2_disc_norm_train)
    h2_disc_gaussian = tf.nn.leaky_relu(A2_disc_gaussian)

    # Output the quotients

    W3_disc = w_variable_mean([ n_units_disc_2, 1 ])
    bias3_disc  = w_variable_mean([ 1 ])

    A3_disc_norm_train = tf.tensordot(h2_disc_norm_train, W3_disc, axes = [[2], [0]]) + bias3_disc
    A3_disc_gaussian = tf.tensordot(h2_disc_gaussian, W3_disc, axes = [[2], [0]]) + bias3_disc

    return A3_disc_norm_train[ :, :, 0 ], A3_disc_gaussian[ :, :, 0 ], [ W1_disc, bias1_disc, W2_disc, bias2_disc, W3_disc, bias3_disc ]


# =============================================================================
# (VAE) Main network
# =============================================================================


##############################
# Case with one hidden layer #
##############################


def exp_log_likelihood(mean_targets, std_targets, weights_train, weights_test, alpha):


    # This is the noise variance

    log_sigma2_noise = tf.Variable(tf.cast(1.0 / 100.0, dtype = tf.float32))

    # Separate the weights and reshape the tensors

    W1_train_re = tf.reshape(weights_train[:,:,:(dim_data * n_units)], shape = [ tf.shape(x)[0], samples_train, n_units, dim_data ])
    W2_train_re  = tf.reshape(weights_train[:,:,(dim_data * n_units):], shape = [ tf.shape(x)[0], samples_train, n_units, 1 ])

    W1_test_re = tf.reshape(weights_test[:,:,:(dim_data * n_units)], shape = [ tf.shape(x)[0], samples_test, n_units, dim_data ])
    W2_test_re  = tf.reshape(weights_test[:,:,(dim_data * n_units):], shape = [ tf.shape(x)[0], samples_test, n_units, 1 ])

    ########################
    #   Processing layer   #
    ########################

    bias_A1 = w_variable_mean([ n_units ])

    A1 = tf.reduce_sum(tf.reshape(x, shape = [tf.shape(x)[0], 1, 1, dim_data]) * W1_train_re, axis = 3) + bias_A1
    A1_test = tf.reduce_sum(tf.reshape(x, shape = [tf.shape(x)[0], 1, 1, dim_data]) * W1_test_re, axis = 3) + bias_A1

    # Results of the layer

    h1 = tf.nn.leaky_relu(A1)
    h1_test = tf.nn.leaky_relu(A1_test)

    ##############################
    #     Regression output      #
    ##############################

    bias_A2  = w_variable_mean([ 1 ])

    A2 = tf.reduce_sum(tf.reshape(h1, shape = [ tf.shape(x)[0], samples_train, n_units, 1 ]) * W2_train_re, axis = 2) + bias_A2
    A2_test = tf.reduce_sum(tf.reshape(h1_test, shape = [ tf.shape(x)[0], samples_test, n_units, 1 ]) * W2_test_re, axis = 2) + bias_A2

    res_train = ( 1.0/alpha) * (- np.log( samples_train ) + tf.reduce_logsumexp( alpha * (-0.5 * (np.log(2.0 * np.pi) + log_sigma2_noise + (A2 - tf.reshape(y_, shape = [ tf.shape(x)[0], 1, 1 ]))**2 / tf.exp(log_sigma2_noise))) , axis = [ 1 ]))

    # This is to compute the test log loglikelihood

    log_prob_data_test = tf.reduce_sum(((tf.reduce_logsumexp(-0.5 * tf.log(2.0 * np.pi * (tf.exp(log_sigma2_noise) * std_targets**2)) \
        - 0.5 * (A2_test * std_targets + mean_targets - tf.reshape(y_, shape = [ tf.shape(x)[0], 1, 1 ]))**2 / (tf.exp(log_sigma2_noise) * std_targets**2), axis = [ 1 ])) \
        - np.log(samples_test)))

    squared_error = tf.reduce_sum((tf.reduce_mean(A2_test, axis = [ 1 ]) * std_targets + mean_targets - y_ )**2)

    return res_train, [ log_sigma2_noise, bias_A1, bias_A2 ], squared_error, log_prob_data_test


###############################
# Case with two hidden layers #
###############################


def exp_log_likelihood_double(mean_targets, std_targets, weights_train, weights_test, alpha):

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

    res_train = ( 1.0/alpha) * (- np.log( samples_train ) + tf.reduce_logsumexp( alpha * (-0.5 * (np.log(2.0 * np.pi) + log_sigma2_noise + (A3 - tf.reshape(y_, shape = [ tf.shape(x)[0], 1, 1 ]))**2 / tf.exp(log_sigma2_noise))) , axis = [ 1 ]))

    # This is to compute the test log loglikelihood

    log_prob_data_test = tf.reduce_sum(((tf.reduce_logsumexp(-0.5 * tf.log(2.0 * np.pi * (tf.exp(log_sigma2_noise) * std_targets**2)) \
        - 0.5 * (A3_test * std_targets + mean_targets - tf.reshape(y_, shape = [ tf.shape(x)[0], 1, 1 ]))**2 / (tf.exp(log_sigma2_noise) * std_targets**2), axis = [ 1 ])) \
        - np.log(samples_test)))

    squared_error = tf.reduce_sum((tf.reduce_mean(A3_test, axis = [ 1 ]) * std_targets + mean_targets - y_ )**2)

    return res_train, [ log_sigma2_noise, bias_A1, bias_A2, bias_A3 ], squared_error, log_prob_data_test

###############################################################################
###############################################################################
###############################################################################


def main(alpha, layers):

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

    data_size = X.shape[ 0 ]
    size_train = data_size - n_size_test
    total_training_data = size_train

    X_train = X[0 : size_train, : ]
    y_train = np.vstack(y[ 0 : size_train ])
    X_test = X[size_train : data_size, : ]
    y_test = np.vstack(y[ size_train : data_size ])

    # Normalize the values

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

    # Obtain the moments of the weights and pass the values through the disc
    mean_w_train, var_w_train = tf.nn.moments(weights_train, axes = [0, 1])

    mean_w_train = tf.stop_gradient(mean_w_train)
    var_w_train = tf.stop_gradient(var_w_train)

    # Normalize real weights

    norm_weights_train = (weights_train - mean_w_train) / tf.sqrt(var_w_train)

    # Generate samples of a normal distribution with the moments of the weights
    if layers == 1:
        w_sampled_gaussian = tf.random_normal(shape = [ tf.shape(x)[0], samples_train, total_number_weights ], mean = 0, stddev = 1, seed = 123)
    elif layers == 2:
        w_sampled_gaussian = tf.random_normal(shape = [ tf.shape(x)[0], samples_train, total_number_weights_double ], mean = 0, stddev = 1, seed = 123)

    # Obtain the T(z,x) for the real and the sampled weights

    T_real, T_sampled, weights_disc = discriminator(norm_weights_train, w_sampled_gaussian, layers)

    # Calculate the cross entropy loss for the discriminator

    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=T_real, labels=tf.ones_like(T_real)))
    d_loss_sampled = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=T_sampled, labels=tf.zeros_like(T_sampled)))

    cross_entropy = d_loss_real + d_loss_sampled

    # Obtain the KL and ELBO

    logr = -0.5 * tf.reduce_sum(norm_weights_train**2 + tf.log(var_w_train) + np.log(2*np.pi), [ 2 ])

    log_vars_prior = w_variable_variance([ 1 ])

    logz = -0.5 * tf.reduce_sum((weights_train)**2 / tf.exp(log_vars_prior) + log_vars_prior + np.log(2*np.pi), [ 2 ])

    KL = T_real + logr - logz

    # Call the main network to calculate the error metrics for the primary system
    if layers == 1:
        res_train, vars_network, squared_error, log_prob_data_test = exp_log_likelihood(mean_targets, std_targets, weights_train, weights_test, alpha) # main loss in the VAE
    elif layers == 2:
        res_train, vars_network, squared_error, log_prob_data_test = exp_log_likelihood_double(mean_targets, std_targets, weights_train, weights_test, alpha) # main loss in the VAE

    # Make the estimates of the ELBO for the primary classifier
    ELBO = tf.reduce_sum(res_train) - tf.reduce_mean(KL) * tf.cast(tf.shape(x)[ 0 ], tf.float32) / tf.cast(total_training_data, tf.float32)

    neg_ELBO = -ELBO
    main_loss = neg_ELBO
    mean_ELBO = ELBO

    # KL y res_train have shape batch_size x n_samples

    mean_KL = tf.reduce_mean(KL)

    # Create the variable lists to be updated

    vars_primal = vars_gen + [ log_vars_prior ] + vars_network
    vars_dual = weights_disc

    train_step_primal = tf.train.AdamOptimizer(primal_rate).minimize(main_loss, var_list = vars_primal)
    train_step_dual = tf.train.AdamOptimizer(dual_rate).minimize(cross_entropy, var_list = vars_dual)

    # Calculate the squared error

    config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1, allow_soft_placement=True, device_count = {'CPU': 1})

    with tf.Session(config = config) as sess:

        sess.run(tf.global_variables_initializer())
        total_ini = time.time()

        # Change the value of alpha to begin exploring using the second value given
        for epoch in range(n_epochs):


            for i_batch in range(int(np.ceil(size_train / n_batch))):

                L = 0.0

                ini = time.clock()
                ini_ref = time.time()
                ini_train = time.clock()

                last_point = np.minimum(n_batch * (i_batch + 1), size_train)

                batch = [ X_train[ i_batch * n_batch : last_point, : ] , y_train[ i_batch * n_batch : last_point, ] ]

                sess.run(train_step_dual, feed_dict={x: batch[0], y_: batch[1]})
                sess.run(train_step_primal, feed_dict={x: batch[0], y_: batch[1]})

                value = sess.run(mean_ELBO, feed_dict={x: batch[0], y_: batch[1]})
                kl = sess.run(mean_KL, feed_dict={x: batch[0], y_: batch[1]})
                L += value

                fini_train = time.clock()

                if i_batch % sampling_batches == 0:

                    ini_test = time.time()

                    # We do the test evaluation RMSE
                    SE = 0.0
                    for i in range(int(np.ceil(X_test.shape[ 0 ] / n_batch))):

                        last_point = np.minimum(n_batch * (i + 1), X_test.shape[ 0 ])

                        batch = [ X_test[ i * n_batch : last_point, : ] , y_test[ i * n_batch : last_point, ] ]

                        #import pdb; pdb.set_trace()

                        SE += sess.run(squared_error, feed_dict={x: batch[0], y_: batch[1]}) / batch[ 0 ].shape[ 0 ]

                    RMSE = np.sqrt(SE / int(np.ceil(X_test.shape[ 0 ] / n_batch)))

                    # We do the test evaluation RMSE
                    LL = 0.0
                    for j in range(int(np.ceil(X_test.shape[ 0 ] / n_batch))):

                        last_point = np.minimum(n_batch * (j + 1), X_test.shape[ 0 ])

                        batch = [ X_test[ j * n_batch : last_point, : ] , y_test[ j * n_batch : last_point, ] ]

                        LL += sess.run(log_prob_data_test, feed_dict={x: batch[0], y_: batch[1]}) / batch[ 0 ].shape[ 0 ]

                    TestLL = (LL / int(np.ceil(X_test.shape[ 0 ] / n_batch)))

                    fini_test = time.time()
                    fini = time.clock()
                    fini_ref = time.time()
                    total_fini = time.time()

                    with open("results_AADM_airlines/res_alpha_" + str(alpha) + "_airlines_350.txt", "a") as res_file:
                       res_file.write('alpha %g batch %g datetime %s epoch %d ELBO %g KL %g real_time %g cpu_time %g train_time %g test_time %g total_time %g LL %g RMSE %g' % (alpha, i_batch, str(datetime.now()), epoch, L, kl, (fini_ref - ini_ref), (fini - ini), (fini_train - ini_train), (fini_test - ini_test), (total_fini - total_ini), TestLL, RMSE) + "\n")


if __name__ == '__main__':


    if not os.path.isdir("results_AADM_airlines"):
        os.makedirs("results_AADM_airlines")

    alpha = np.float(sys.argv[1])
    layers = int(sys.argv[2])

    if os.path.isfile("results_AADM_airlines/res_alpha_" + str(alpha) + "_airlines.txt"):
        with open("results_AADM_airlines/res_alpha_" + str(alpha) + "_airlines.txt", "w") as file:
            file.close()


    main(alpha, layers)
