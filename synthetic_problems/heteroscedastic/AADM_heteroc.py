###############################################################################
############## ADVERSARIAL ALPHA DIVERGENCE MINIMIZATION WITH NN ##############
###############################################################################
#
# This code performs AADM with neural networks in a synthetic toy problem of
# bimodally distributed data
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

import tensorflow as tf
import numpy as np

import os
os.chdir(".")

# =============================================================================
# Complete system parameters (VAE, encoder, decoder, data specs, etc.)
# =============================================================================

tf.set_random_seed(123)

# File to be analyzed
original_file = "data_heteroc.txt"

# This is the total number of training samples
total_training_data = 1.0
n_batch = 10
n_epochs = 1500
n_epochs_change = 3000 # int((1/2)*n_epochs) # Uncomment this to employ the first half of the training with alpha = 1.0 as warming-up period

mean_targets = 0.0
std_targets = 1.0

# Number of samples for the weights in each layer
samples_train = 10
samples_test = 50
ratio = 0.5 # Portion of the data in the train set


# Structural parameters of the main NN
dim_data = 1
n_units = 50
n_units_sec = 50    # For the case with 2 hidden layers
total_number_weights = (dim_data + 1) * n_units  # Total number of weights used
total_number_weights_double = n_units * (dim_data + n_units_sec) + n_units_sec  # Number of weights for the 2 hidden layers case

# Parameters for the generator network
n_units_gen = 350         # Number of units in the generator for 1 hidden layer in the VAE
n_units_gen_double = 400  # Number of units in the generator for 2 hidden layers in the VAE
noise_comps_gen = 100     # Number of gaussian variables inputed in the encoder

# Parameters for the discriminative network
n_units_disc = 350

# Learning ratios
primal_rate = 1e-4
dual_rate = 1e-3

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
########################### Network structure #################################
###############################################################################

# =============================================================================
#  (Encoder) Deterministic NN that generates the weights
# =============================================================================

##############################
# Case with one hidden layer #
##############################


def generate_weights(batchsize, samples_train, samples_test):

    # Inputs
    # batchsize :   Dimension of the batch of data we are going to apply the weights to
    # samples   :   Number of samples of the weights required
    #

    # Initialize the random gaussian noises to input the network

    init_noise_train  = tf.random_normal(shape = [ batchsize, samples_train, noise_comps_gen ])
    init_noise_test = tf.random_normal(shape = [ batchsize, samples_test, noise_comps_gen ])

    #Process the noises through the network
    W1_gen = w_variable_mean([ noise_comps_gen, n_units_gen ])
    bias1_gen  = w_variable_mean([ n_units_gen ])

    A1_gen_train = tf.tensordot(init_noise_train, W1_gen, axes = [[2], [0]]) + bias1_gen
    A1_gen_test = tf.tensordot(init_noise_test, W1_gen, axes = [[2], [0]]) + bias1_gen

    # Results for the layer
    h1_gen_train = tf.nn.leaky_relu(A1_gen_train)
    h1_gen_test = tf.nn.leaky_relu(A1_gen_test)

    # Output the weights
    W2_gen = w_variable_mean([ n_units_gen, (dim_data + 1) * n_units ])
    bias2_gen  = w_variable_mean([ (dim_data + 1) * n_units ])

    A2_gen_train = tf.tensordot(h1_gen_train, W2_gen, axes = [[2], [0]]) + bias2_gen    # final weights
    A2_gen_test = tf.tensordot(h1_gen_test, W2_gen, axes = [[2], [0]]) + bias2_gen    # final weights

    # The dimensions here are (batchsize, samples, (dim_data + 1) * n_units), and
    # therefore, we can allocate the weights of the different parts of the main NN

    return A2_gen_train, A2_gen_test, [ W1_gen, bias1_gen, W2_gen, bias2_gen ]


###############################
# Case with two hidden layers #
###############################


def generate_weights_double(batchsize, samples_train, samples_test):

    # Inputs
    # batchsize :   Dimension of the batch of data we are going to apply the weights to
    # samples   :   Number of samples of the weights required
    #

    # Initialize the random gaussian noises to input the network
    init_noise_train  = tf.random_normal(shape = [ batchsize, samples_train, noise_comps_gen ])
    init_noise_test = tf.random_normal(shape = [ batchsize, samples_test, noise_comps_gen ])

    #Process the noises through the network
    W1_gen = w_variable_mean([ noise_comps_gen, n_units_gen_double ])
    bias1_gen  = w_variable_mean([ n_units_gen_double ])

    A1_gen_train = tf.tensordot(init_noise_train, W1_gen, axes = [[2], [0]]) + bias1_gen
    A1_gen_test = tf.tensordot(init_noise_test, W1_gen, axes = [[2], [0]]) + bias1_gen

    # Results for the layer
    h1_gen_train = tf.nn.leaky_relu(A1_gen_train)
    h1_gen_test = tf.nn.leaky_relu(A1_gen_test)

    # Output the weights
    W2_gen = w_variable_mean([ n_units_gen_double, total_number_weights_double ])
    bias2_gen  = w_variable_mean([total_number_weights_double])

    A2_gen_train = tf.tensordot(h1_gen_train, W2_gen, axes = [[2], [0]]) + bias2_gen    # final weights
    A2_gen_test = tf.tensordot(h1_gen_test, W2_gen, axes = [[2], [0]]) + bias2_gen    # final weights


    return A2_gen_train, A2_gen_test, [ W1_gen, bias1_gen, W2_gen, bias2_gen ]


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

    # Results for the layer

    h1_disc_norm_train = tf.nn.leaky_relu(A1_disc_norm_train)
    h1_disc_gaussian = tf.nn.leaky_relu(A1_disc_gaussian)

    # Output the quotients

    W2_disc = w_variable_mean([ n_units_disc, 1 ])
    bias2_disc  = w_variable_mean([ 1 ])

    A2_disc_norm_train = tf.tensordot(h1_disc_norm_train, W2_disc, axes = [[2], [0]]) + bias2_disc
    A2_disc_gaussian = tf.tensordot(h1_disc_gaussian, W2_disc, axes = [[2], [0]]) + bias2_disc

    return A2_disc_norm_train[ :, :, 0 ], A2_disc_gaussian[ :, :, 0 ], [ W1_disc, bias1_disc, W2_disc, bias2_disc ]



# =============================================================================
# (VAE) Main network
# =============================================================================


##############################
# Case with one hidden layer #
##############################


def exp_log_likelihood(mean_targets, std_targets, weights_train, weights_test, alpha, alpha_2):


    # This is the noise variance

    log_sigma2_noise = tf.Variable(tf.cast(1.0 / 100.0, dtype = tf.float32))

    # Separate the weights and reshape the tensors

    W1_train_re = tf.reshape(weights_train[:,:,:(dim_data * n_units)], shape = [ tf.shape(x)[0], samples_train, n_units, dim_data ])
    W2_train_re  = tf.reshape(weights_train[:,:,(dim_data * n_units):], shape = [ tf.shape(x)[0], samples_train, n_units, 1 ])

    W1_test_re = tf.reshape(weights_test[:,:,:(dim_data * n_units)], shape = [ tf.shape(x)[0], samples_test, n_units, dim_data ])
    W2_test_re  = tf.reshape(weights_test[:,:,(dim_data * n_units):], shape = [ tf.shape(x)[0], samples_test, n_units, 1 ])

    #########################
    # Capa de procesamiento #
    #########################

    bias_A1 = w_variable_mean([ n_units ])

    A1 = tf.reduce_sum(tf.reshape(x, shape = [tf.shape(x)[0], 1, 1, dim_data]) * W1_train_re, axis = 3) + bias_A1
    A1_test = tf.reduce_sum(tf.reshape(x, shape = [tf.shape(x)[0], 1, 1, dim_data]) * W1_test_re, axis = 3) + bias_A1

    # Results of the layer

    h1 = tf.nn.leaky_relu(A1)
    h1_test = tf.nn.leaky_relu(A1_test)

    ##############################
    # Capa de salida regresion   #
    ##############################

    bias_A2  = w_variable_mean([ 1 ])

    A2 = tf.reduce_sum(tf.reshape(h1, shape = [ tf.shape(x)[0], samples_train, n_units, 1 ]) * W2_train_re, axis = 2) + bias_A2
    A2_test = tf.reduce_sum(tf.reshape(h1_test, shape = [ tf.shape(x)[0], samples_test, n_units, 1 ]) * W2_test_re, axis = 2) + bias_A2

    res_train = ( 1.0/alpha) * (- np.log( samples_train ) + tf.reduce_logsumexp( alpha * (-0.5 * (np.log(2.0 * np.pi) + log_sigma2_noise + (A2 - tf.reshape(y_, shape = [ tf.shape(x)[0], 1, 1 ]))**2 / tf.exp(log_sigma2_noise))) , axis = [ 1 ]))
    res_train_2 = ( 1.0/alpha_2) * (- np.log( samples_train ) + tf.reduce_logsumexp( alpha_2 * (-0.5 * (np.log(2.0 * np.pi) + log_sigma2_noise + (A2 - tf.reshape(y_, shape = [ tf.shape(x)[0], 1, 1 ]))**2 / tf.exp(log_sigma2_noise))) , axis = [ 1 ]))

    # This is to compute the test log loglikelihood

    log_prob_data_test = tf.reduce_sum(((tf.reduce_logsumexp(-0.5 * tf.log(2.0 * np.pi * (tf.exp(log_sigma2_noise) * std_targets**2)) \
        - 0.5 * (A2_test * std_targets + mean_targets - tf.reshape(y_, shape = [ tf.shape(x)[0], 1, 1 ]))**2 / (tf.exp(log_sigma2_noise) * std_targets**2), axis = [ 1 ])) \
        - np.log(samples_test)))

    squared_error = tf.reduce_sum((tf.reduce_mean(A2_test, axis = [ 1 ]) * std_targets + mean_targets - y_ )**2)

    y_test  = tf.random_normal(shape = [ tf.shape(x)[0], samples_test ])*tf.exp(log_sigma2_noise) + A2_test[:,:,0]
    y_test = y_test * std_targets + mean_targets

    return res_train, res_train_2, [ log_sigma2_noise, bias_A1, bias_A2 ], squared_error, log_prob_data_test, y_test


###############################
# Case with two hidden layers #
###############################


def exp_log_likelihood_double(mean_targets, std_targets, weights_train, weights_test, alpha, alpha_2):


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
    res_train_2 = ( 1.0/alpha_2) * (- np.log( samples_train ) + tf.reduce_logsumexp( alpha_2 * (-0.5 * (np.log(2.0 * np.pi) + log_sigma2_noise + (A3 - tf.reshape(y_, shape = [ tf.shape(x)[0], 1, 1 ]))**2 / tf.exp(log_sigma2_noise))) , axis = [ 1 ]))

    # This is to compute the test log loglikelihood

    log_prob_data_test = tf.reduce_sum(((tf.reduce_logsumexp(-0.5 * tf.log(2.0 * np.pi * (tf.exp(log_sigma2_noise) * std_targets**2)) \
        - 0.5 * (A3_test * std_targets + mean_targets - tf.reshape(y_, shape = [ tf.shape(x)[0], 1, 1 ]))**2 / (tf.exp(log_sigma2_noise) * std_targets**2), axis = [ 1 ])) \
        - np.log(samples_test)))

    squared_error = tf.reduce_sum((tf.reduce_mean(A3_test, axis = [ 1 ]) * std_targets + mean_targets - y_ )**2)

    y_test_pre  = tf.random_normal(shape = [ tf.shape(x)[0], samples_test ])*tf.exp(log_sigma2_noise) + A3_test[:,:,0]
    y_test = y_test_pre * std_targets + mean_targets

    return res_train, res_train_2, [ log_sigma2_noise, bias_A1, bias_A2, bias_A3 ], squared_error, log_prob_data_test, y_test, y_test_pre, A3_test

###############################################################################
###############################################################################
###############################################################################


def main(alpha, alpha_2, layers):

    np.random.seed(1234)
    tf.set_random_seed(1234)

    # We load the original dataset
    data = np.loadtxt(original_file)

    # =========================================================================
    #  Parameters of the complete system
    # =========================================================================

    # We obtain the features and the targets

    X = data[ :, range(data.shape[ 1 ] - 1) ]
    y = data[ :, data.shape[ 1 ] - 1 ]

    # We create the train and test sets with 100*ratio % of the data

    size_train = int(np.round(X.shape[ 0 ] * ratio))
    total_training_data = size_train

    X_train = X[0 : size_train,]
    y_train = np.vstack(y[ 0 : size_train ])
    X_test = X[size_train : X.shape[0],]
    y_test = np.vstack(y[ size_train : y.shape[0] ])

    # Normalizamos los argumentos

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
        w_sampled_gaussian = tf.random_normal(shape = [ tf.shape(x)[0], samples_train, total_number_weights ], mean = 0, stddev = 1)
    elif layers == 2:
        w_sampled_gaussian = tf.random_normal(shape = [ tf.shape(x)[0], samples_train, total_number_weights_double ], mean = 0, stddev = 1)

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
        res_train, res_train_2, vars_network, squared_error, log_prob_data_test, res_test = exp_log_likelihood(mean_targets, std_targets, weights_train, weights_test, alpha, alpha_2) # main loss in the VAE
    elif layers == 2:
        res_train, res_train_2, vars_network, squared_error, log_prob_data_test, res_test, y_test_pre, A3_test = exp_log_likelihood_double(mean_targets, std_targets, weights_train, weights_test, alpha, alpha_2) # main loss in the VAE

    # Make the estimates of the ELBO for the primary classifier
    ELBO = tf.reduce_sum(res_train) - tf.reduce_mean(KL) * tf.cast(tf.shape(x)[ 0 ], tf.float32) / tf.cast(total_training_data, tf.float32)
    ELBO_2 = tf.reduce_sum(res_train_2) - tf.reduce_mean(KL) * tf.cast(tf.shape(x)[ 0 ], tf.float32) / tf.cast(total_training_data, tf.float32)

    neg_ELBO = -ELBO
    main_loss = neg_ELBO
    mean_ELBO = ELBO

    neg_ELBO_2 = -ELBO_2
    main_loss_2 = neg_ELBO_2
    mean_ELBO_2 = ELBO_2

    # KL y res_train have shape batch_size x n_samples

    mean_KL = tf.reduce_mean(KL)

    # Create the variable lists to be updated

    vars_primal = vars_gen + [ log_vars_prior ] + vars_network
    vars_dual = weights_disc

    train_step_primal = tf.train.AdamOptimizer(primal_rate).minimize(main_loss, var_list = vars_primal)
    train_step_primal_2 = tf.train.AdamOptimizer(primal_rate).minimize(main_loss_2, var_list = vars_primal)

    train_step_dual = tf.train.AdamOptimizer(dual_rate).minimize(cross_entropy, var_list = vars_dual)

    # Calculate the squared error

    timing = list()
    tmp_kl = list()
    tmp_elbo = list()
    results = list()

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        # Change the value of alpha to begin exploring using the second value given
        for epoch in range(n_epochs):

            permutation = np.random.choice(range(X_train.shape[ 0 ]), X_train.shape[ 0 ], replace = False)
            X_train = X_train[ permutation, : ]
            y_train = y_train[ permutation, : ]

            if epoch < n_epochs_change:
                primal_metric_train = train_step_primal
            elif epoch >= n_epochs_change:
                alpha = alpha_2
                primal_metric_train = train_step_primal_2

            L = 0.0

            ini = time.clock()
            ini_ref = time.time()

            for i in range(int(np.ceil(X_train.shape[ 0 ] / n_batch))):

                last_point = np.minimum(n_batch * (i + 1), X_train.shape[ 0 ])

                batch = [ X_train[ i * n_batch : last_point, : ] , y_train[ i * n_batch : last_point, ] ]

                sess.run(train_step_dual, feed_dict={x: batch[0], y_: batch[1]})
                sess.run(primal_metric_train, feed_dict={x: batch[0], y_: batch[1]})

                value = sess.run(mean_ELBO, feed_dict={x: batch[0], y_: batch[1]})
                kl = sess.run(mean_KL, feed_dict={x: batch[0], y_: batch[1]})
                L += value

            fini = time.clock()
            fini_ref = time.time()
            if epoch % 100 == 0:
                print('alpha %g; epoch %d; ELBO %g; real_time %g; cpu_time %g; KL %g' % (alpha, epoch, L, (fini_ref - ini_ref), (fini - ini),  kl))


            timing.append(fini-ini)
            tmp_kl.append(kl)
            tmp_elbo.append(L)


            sys.stdout.flush()

        results_write = sess.run(res_test, feed_dict={x: X_test, y_: y_test})

        # We do the test evaluation RMSE
        SE = 0.0
        for i in range(int(np.ceil(X_test.shape[ 0 ] / n_batch))):

            last_point = np.minimum(n_batch * (i + 1), X_test.shape[ 0 ])

            batch = [ X_test[ i * n_batch : last_point, : ] , y_test[ i * n_batch : last_point, ] ]

            SE += sess.run(squared_error, feed_dict={x: batch[0], y_: batch[1]}) / batch[ 0 ].shape[ 0 ]

        RMSE = np.sqrt(SE / int(np.ceil(X_test.shape[ 0 ] / n_batch)))

        print('RMSE %g' % RMSE)

        # We do the test evaluation RMSE

        LL = 0.0
        for i in range(int(np.ceil(X_test.shape[ 0 ] / n_batch))):

            last_point = np.minimum(n_batch * (i + 1), X_test.shape[ 0 ])

            batch = [ X_test[ i * n_batch : last_point, : ] , y_test[ i * n_batch : last_point, ] ]

            LL += sess.run(log_prob_data_test, feed_dict={x: batch[0], y_: batch[1]}) / batch[ 0 ].shape[ 0 ]

        TestLL = (LL / int(np.ceil(X_test.shape[ 0 ] / n_batch)))

        print('Test log. likelihood %g' % TestLL)

        np.savetxt('res_bim/' + str(alpha_2) + 'results_rmse.txt', [ RMSE ])
        np.savetxt('res_bim/' + str(alpha_2) + 'results_ll.txt', [ TestLL ])
        np.savetxt('res_bim/' + str(alpha_2) + 'results_time.txt', [ timing ])
        np.savetxt('res_bim/' + str(alpha_2) + 'results_KL.txt', [ tmp_kl ])
        np.savetxt('res_bim/' + str(alpha_2) + 'results_ELBO.txt', [ tmp_elbo ])
        np.savetxt('res_bim/' + str(alpha_2) + 'results_ytest.txt', results_write)
        np.savetxt('res_bim/' + str(alpha_2) + 'results_xtest.txt', X_test)


if __name__ == '__main__':


    if not os.path.isdir("res_bim"):
        os.makedirs("res_bim")

    # We load the permutation we want to do the calculations in
    # available_perm = np.loadtxt('permutations_' + original_file, delimiter = ",", dtype = int)


    # split = int(sys.argv[1])
    alpha = float(sys.argv[1])
    alpha_2 = float(sys.argv[2])
    layers = int(sys.argv[3])

    main(alpha, alpha_2, layers)

