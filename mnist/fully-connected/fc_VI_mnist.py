########################################################################
######## VARIATIONAL INFERENCE in  MNIST with alpha divergences ########
########################################################################

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

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)

import os
os.chdir(".")

seed = 123

# =============================================================================
# DATA PARAMETERS AND SYSTEM ARCHITECTURE
# =============================================================================

batches_to_report = 300
# File to be analyzed
original_file = "mnist"

# This is the total number of training samples
n_batch = 200
n_batch_test = 1    # Since we will need more samples in test, we make the batchsize smaller to keep everything in manageable sizes
num_classes = 10
n_epochs = 1000

# input image dimensions
img_width, img_height = 28, 28
channels_img = 1

n_samples_train = 10
n_samples_test = 750

# Structural parameters of the NN system
n_layers_gen = 2          # Number of layers in the generator
n_units_gen = 200          # Number of units in the generator for 1 hidden layer in the VAE
noise_comps_gen = 100     # Number of gaussian variables inputed in the encoder

# Parameters for the discriminative network
n_units_disc = 200
n_layers_disc = 2

# Learning rates
primal_rate = 1e-3 # Actual Bayesian NN
dual_rate = 1e-3   # Discriminator

# ==============================================================================
# DEFINE HERE THE STRUCTURE OF THE CNN
# ==============================================================================

n_units_dense_1 = 200
n_units_dense_2 = 200
n_classes = 10

# FULLY CONNECTED CASE

dim_data = img_width * img_height * channels_img

dict_dim_weights = {
    'd1': [dim_data, n_units_dense_1],
    'd2': [n_units_dense_1, n_units_dense_2],
    'd3': [n_units_dense_2, n_classes]
}

dict_dim_biases = {
    'd1': n_units_dense_1, 'd2': n_units_dense_2,       # Convolution Layers
    'd3': n_classes,    # Dense Layers
}

# We define the following auxiliary functions to simplify the rest of the code

def w_variable_mean(shape):
  initial = tf.random_normal(shape = shape, stddev = 0.001, seed = seed) # mean 0 stddev 1
  return tf.Variable(initial)

def w_variable_variance(shape):
  initial = tf.random_normal(shape = shape, stddev = 0.001, seed = seed) - 10.0 # mean 0 stddev 1
  return tf.Variable(initial)

def one_hot(a, num_classes):
  return np.squeeze(np.eye(num_classes)[a.reshape(-1)])


###############################################################################
##########################   Network structure  ###############################
###############################################################################

# NF Elliminate all the references to the batchsize dimension in the generator functions

def create_generator(n_units_gen, noise_comps_gen, total_number_z, n_layers_gen):

    mean_noise = w_variable_mean([ 1, noise_comps_gen ])
    log_var_noise = w_variable_variance([ 1, noise_comps_gen ])

    W1_gen = w_variable_mean([ noise_comps_gen, n_units_gen ])
    bias1_gen  = w_variable_mean([ n_units_gen ])

    W2_gen = w_variable_mean([ n_units_gen, n_units_gen ])
    bias2_gen  = w_variable_mean([ n_units_gen ])

    W3_gen = w_variable_mean([ n_units_gen, total_number_z ])
    bias3_gen  = w_variable_mean([ total_number_z ])

    return {'mean_noise': mean_noise, 'log_var_noise': log_var_noise, 'W1_gen': W1_gen, 'bias1_gen': bias1_gen, \
        'W2_gen': W2_gen, 'bias2_gen': bias2_gen, 'W3_gen': W3_gen, 'bias3_gen': bias3_gen, \
        'n_layers_gen': n_layers_gen}

def get_variables_generator(generator):
    return [ generator['mean_noise'], generator['log_var_noise'], generator['W1_gen'], \
        generator['W2_gen'], generator['W3_gen'], generator['bias1_gen'], generator['bias2_gen'], generator['bias3_gen'] ]

def compute_output_generator(generator, n_samples, noise_comps_gen):

    mean_noise = generator['mean_noise']
    log_var_noise = generator['log_var_noise']

    W1_gen = generator['W1_gen']
    W2_gen = generator['W2_gen']
    W3_gen = generator['W3_gen']

    bias1_gen = generator['bias1_gen']
    bias2_gen = generator['bias2_gen']
    bias3_gen = generator['bias3_gen']

    pre_init_noise = tf.random_normal(shape = [ n_samples, noise_comps_gen ], seed = seed)

    init_noise =  mean_noise + tf.sqrt(tf.exp(log_var_noise)) * pre_init_noise

    # Process the noises through the network

    A1_gen = tf.tensordot(init_noise, W1_gen, axes = [[1], [0]]) + bias1_gen
    h1_gen = tf.nn.leaky_relu(A1_gen)

    if generator['n_layers_gen'] == 1:
        A3_gen = tf.tensordot(h1_gen, W3_gen, axes = [[1], [0]]) + bias3_gen    # final weights
    else:
        A2_gen = tf.tensordot(h1_gen, W2_gen, axes = [[1], [0]]) + bias2_gen
        h2_gen = tf.nn.leaky_relu(A2_gen)
        A3_gen = tf.tensordot(h2_gen, W3_gen, axes = [[1], [0]]) + bias3_gen    # final weights

    return A3_gen * 0.0 + 1.0

def create_discriminator(n_units_disc, total_weights, n_layers_disc):

    W1_disc = w_variable_mean([ total_weights, n_units_disc])
    bias1_disc = w_variable_mean([ n_units_disc ])

    W2_disc = w_variable_mean([ n_units_disc, n_units_disc ])
    bias2_disc = w_variable_mean([ n_units_disc ])

    W3_disc = w_variable_mean([ n_units_disc ,  1 ])
    bias3_disc = w_variable_mean([ 1 ])

    return {'W1_disc': W1_disc, 'bias1_disc': bias1_disc, 'W2_disc': W2_disc, \
        'bias2_disc': bias2_disc, 'W3_disc': W3_disc, 'bias3_disc': bias3_disc, 'n_layers_disc': n_layers_disc,
        'total_weights': total_weights}

def get_variables_discriminator(discriminator):
    return [ discriminator['W1_disc'], discriminator['W2_disc'], discriminator['W3_disc'], discriminator['bias1_disc'], \
        discriminator['bias2_disc'], discriminator['bias3_disc'] ]

def compute_output_discriminator(discriminator, weights, n_layers):

    total_weights = discriminator['total_weights']
    W1_disc = discriminator['W1_disc']
    W2_disc = discriminator['W2_disc']
    W3_disc = discriminator['W3_disc']

    bias1_disc = discriminator['bias1_disc']
    bias2_disc = discriminator['bias2_disc']
    bias3_disc = discriminator['bias3_disc']

    A1_disc = tf.matmul(weights, W1_disc) + bias1_disc
    h1_disc = tf.nn.leaky_relu(A1_disc)

    if discriminator['n_layers_disc'] == 2:
        A2_disc = tf.matmul(h1_disc, W2_disc) + bias2_disc
        h2_disc = tf.nn.leaky_relu(A2_disc)

        A3_disc = tf.matmul(h2_disc, W3_disc) + bias3_disc
    else:
        A3_disc = tf.matmul(h1_disc, W3_disc) + bias3_disc

    return A3_disc[ :, 0 ]

# ==============================================================================
# CONVOLUTIONAL NN MAIN
# ==============================================================================

def create_main_NN(dict_weights, dict_biases):

    # The biases are not random

    bias_D1 = w_variable_mean([ dict_biases["d1"] ])
    bias_D2 = w_variable_mean([ dict_biases["d2"] ])
    bias_D3 = w_variable_mean([ dict_biases["d3"] ])

    # NF
    # Create variable for the means and variances needed

    mean_D1 = w_variable_mean(dict_weights["d1"])
    mean_D2 = w_variable_mean(dict_weights["d2"])
    mean_D3 = w_variable_mean(dict_weights["d3"])

    log_sigma2_D1 = w_variable_variance(dict_weights["d1"])
    log_sigma2_D2 = w_variable_variance(dict_weights["d2"])
    log_sigma2_D3 = w_variable_variance(dict_weights["d3"])

    return {'bias_D1': bias_D1, 'bias_D2': bias_D2, 'bias_D3': bias_D3, \
        'mean_D1': mean_D1, 'mean_D2': mean_D2, 'mean_D3': mean_D3, \
        'log_sigma2_D1': log_sigma2_D1, 'log_sigma2_D2': log_sigma2_D2, 'log_sigma2_D3': log_sigma2_D3}

def get_variables_main_NN(network):

    return [ network['bias_D1'], network['bias_D2'], network['bias_D3'], \
        network['mean_D1'], network['mean_D2'], network['mean_D3'], \
        network['log_sigma2_D1'], network['log_sigma2_D2'], network['log_sigma2_D3']]

def compute_outputs_main_NN(z_samples, dict_main_NN, dict_dim_weights, x_input, y_target, n_samples, training = 0):

    batch_size = tf.shape(x_input)[ 0 ]

    x_input = tf.reshape(x_input, [ 1, tf.shape(x_input)[ 0 ], tf.shape(x_input)[ 1 ] ])

    bias_1 = dict_main_NN["bias_D1"]
    bias_2 = dict_main_NN["bias_D2"]
    bias_3 = dict_main_NN["bias_D3"]

    means_1 = dict_main_NN["mean_D1"]
    means_2 = dict_main_NN["mean_D2"]
    means_3 = dict_main_NN["mean_D3"]

    log_vars_1 = dict_main_NN["log_sigma2_D1"]
    log_vars_2 = dict_main_NN["log_sigma2_D2"]
    log_vars_3 = dict_main_NN["log_sigma2_D3"]

    z_samples_1 = z_samples[ :, 0 : dict_dim_weights['d1'][ 0 ] ]
    z_samples_1 = tf.reshape(z_samples_1, [ tf.shape(z_samples_1)[ 0 ], 1, tf.shape(z_samples_1)[ 1 ] ])

    z_samples_2 = z_samples[ :, dict_dim_weights['d1'][ 0 ] : (dict_dim_weights['d1'][ 0 ] + dict_dim_weights['d2'][ 0 ]) ]
    z_samples_2 = tf.reshape(z_samples_2, [ tf.shape(z_samples_2)[ 0 ], 1, tf.shape(z_samples_2)[ 1 ] ])

    z_samples_3 = z_samples[ :, (dict_dim_weights['d1'][ 0 ] + dict_dim_weights['d2'][ 1 ]) : (dict_dim_weights['d1'][ 0 ] +\
         dict_dim_weights['d2'][ 0 ] + dict_dim_weights['d3'][ 0 ]) ]
    z_samples_3 = tf.reshape(z_samples_3, [ tf.shape(z_samples_3)[ 0 ], 1, tf.shape(z_samples_3)[ 1 ] ])

    A1_means = tf.tensordot(x_input * z_samples_1, means_1, axes = [[2], [0]])
    A1_vars = tf.tensordot(x_input**2, tf.exp(log_vars_1), axes = [[2], [0]])
    A1 = tf.random_normal(shape = tf.shape(A1_means), seed = seed) * tf.sqrt(A1_vars) + A1_means + bias_1
    h1 = tf.nn.leaky_relu(A1)

    A2_means = tf.tensordot(h1 * z_samples_2, means_2, axes = [[2], [0]])
    A2_vars = tf.tensordot(h1**2, tf.exp(log_vars_2), axes = [[2], [0]])
    A2 = tf.random_normal(shape = tf.shape(A2_means), seed = seed) * tf.sqrt(A2_vars) + A2_means + bias_2
    h2 = tf.nn.leaky_relu(A2)

    A3_means = tf.tensordot(h2 * z_samples_3, means_3, axes = [[2], [0]])
    A3_vars = tf.tensordot(h2**2, tf.exp(log_vars_3), axes = [[2], [0]])
    A3 = tf.random_normal(shape = tf.shape(A3_means), seed = seed) * tf.sqrt(A3_vars) + A3_means + bias_3
    # Done!

    A3 = tf.transpose(A3, [ 1, 0, 2]) # A3 should be n_samples x n_batch x 10 instead of n_batch x n_samples x 10

    #######################
    # Objective functions #
    #######################

    probs = tf.nn.softmax(A3, axis = 2)
    pred_classes = tf.argmax(tf.reduce_mean(tf.nn.softmax(A3, axis = 2), axis = 1), axis = -1)

    tiled_y = tf.tile(tf.expand_dims(y_target, 0), [n_samples, 1, 1])
    tiled_y = tf.transpose(tiled_y, [1, 0, 2])
    labels = tf.argmax(tiled_y, 2)

    # Objective function


    res_train = tf.reduce_mean(-1.0 * tf.nn.sparse_softmax_cross_entropy_with_logits(labels = labels, logits = A3), axis = 1)

    # Test log loglikelihood

    log_prob_data = tf.reduce_mean(tf.reduce_logsumexp(-1.0 * tf.nn.sparse_softmax_cross_entropy_with_logits(\
        labels = tf.argmax(tiled_y, axis = 2), logits = A3), axis = 1) - tf.log(tf.cast(n_samples, tf.float32)))

    # Classification error

    class_error = 1.0 - tf.reduce_mean(tf.cast(tf.equal(tf.argmax(tf.reduce_mean(tf.nn.softmax(A3, axis = 2), axis = 1), \
        axis = 1), tf.argmax(y_target, axis = 1)), tf.float32))

    acc_op = 1.0 - class_error

    # Multiclass Brier score

    probs_robust = tf.reduce_mean(tf.nn.softmax(A3, axis = -1), axis = 1)
    brier_score =  tf.reduce_mean(tf.reduce_sum((probs_robust - y_target)**2, axis = -1))

    return res_train, log_prob_data, class_error, acc_op, pred_classes, tf.argmax(tiled_y, axis = 2), brier_score

###############################################################################
###############################################################################

# NF
# Create the function to sample the weights from the z_samples and means and variances from the main network

def sample_weights(dict_main_NN, z_samples, dict_dim_weights):

    means_1 = dict_main_NN["mean_D1"]
    means_2 = dict_main_NN["mean_D2"]
    means_3 = dict_main_NN["mean_D3"]

    log_vars_1 = dict_main_NN["log_sigma2_D1"]
    log_vars_2 = dict_main_NN["log_sigma2_D2"]
    log_vars_3 = dict_main_NN["log_sigma2_D3"]

    n_samples = tf.shape(z_samples)[ 0 ]

    z_samples_1 = z_samples[ :, 0 : dict_dim_weights['d1'][ 0 ] ]
    z_samples_1 = tf.reshape(z_samples_1, [ tf.shape(z_samples_1)[ 0 ], tf.shape(z_samples_1)[ 1 ], 1 ])

    z_samples_2 = z_samples[ :, dict_dim_weights['d1'][ 0 ] : (dict_dim_weights['d1'][ 0 ] + dict_dim_weights['d2'][ 0 ]) ]
    z_samples_2 = tf.reshape(z_samples_2, [ tf.shape(z_samples_2)[ 0 ], tf.shape(z_samples_2)[ 1 ], 1 ])

    z_samples_3 = z_samples[ :, (dict_dim_weights['d1'][ 0 ] + dict_dim_weights['d2'][ 1 ]) : (dict_dim_weights['d1'][ 0 ] +\
         dict_dim_weights['d2'][ 0 ] + dict_dim_weights['d3'][ 0 ]) ]
    z_samples_3 = tf.reshape(z_samples_3, [ tf.shape(z_samples_3)[ 0 ], tf.shape(z_samples_3)[ 1 ], 1 ])

    A1_means = tf.reshape(z_samples_1 * means_1, [ n_samples, dict_dim_weights['d1'][ 0 ] * dict_dim_weights['d1'][ 1 ] ])
    A1_vars = tf.exp(tf.reshape(log_vars_1, [ 1, dict_dim_weights['d1'][ 0 ] * dict_dim_weights['d1'][ 1 ] ]))
    W1_samples = tf.random_normal(shape = tf.shape(A1_means), seed = seed) * tf.sqrt(A1_vars) + A1_means

    A2_means = tf.reshape(z_samples_2 * means_2, [ n_samples, dict_dim_weights['d2'][ 0 ] * dict_dim_weights['d2'][ 1 ] ])
    A2_vars = tf.exp(tf.reshape(log_vars_2, [ 1, dict_dim_weights['d2'][ 0 ] * dict_dim_weights['d2'][ 1 ] ]))
    W2_samples = tf.random_normal(shape = tf.shape(A2_means), seed = seed) * tf.sqrt(A2_vars) + A2_means

    A3_means = tf.reshape(z_samples_3 * means_3, [ n_samples, dict_dim_weights['d3'][ 0 ] * dict_dim_weights['d3'][ 1 ] ])
    A3_vars = tf.exp(tf.reshape(log_vars_3, [ 1, dict_dim_weights['d3'][ 0 ] * dict_dim_weights['d3'][ 1 ] ]))
    W3_samples = tf.random_normal(shape = tf.shape(A3_means), seed = seed) * tf.sqrt(A3_vars) + A3_means

    weights = tf.concat((W1_samples, W2_samples, W3_samples), axis = 1)
    weights_means = tf.concat((tf.reduce_mean(A1_means, 0), tf.reduce_mean(A2_means, 0), tf.reduce_mean(A3_means, 0)), axis = 0)
    weights_vars = tf.concat((tf.reduce_mean(A1_vars + A1_means**2, 0) - tf.reduce_mean(A1_means, 0)**2, \
        tf.reduce_mean(A2_vars + A2_means**2, 0) - tf.reduce_mean(A2_means, 0)**2, \
        tf.reduce_mean(A3_vars + A3_means**2, 0) - tf.reduce_mean(A3_means, 0)**2), axis = 0)

    return weights, weights_means, weights_vars

###############################################################################
###############################################################################

def main():

    np.random.seed(seed)
    tf.set_random_seed(seed)

    # We load the original dataset

    # =========================================================================
    #  Inputs for the main system
    # =========================================================================
    # Import the dataset
    # the data, split between train and test sets
    x_train = mnist.train.images
    y_train = mnist.train.labels

    x_train = np.vstack((x_train, mnist.validation.images))
    y_train = np.concatenate((y_train, mnist.validation.labels))

    x_test = mnist.test.images
    y_test = mnist.test.labels

    # (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Flatten the images into vectors

    X_train = x_train.reshape(x_train.shape[0], dim_data)
    X_test = x_test.reshape(x_test.shape[0], dim_data)

    total_training_data = x_train.shape[0]

    # convert class vectors to binary class matrices

    y_train = one_hot(y_train, num_classes)
    y_test = one_hot(y_test, num_classes)

    # Create the model
    # Placeholders for data and number of samples

    x = tf.placeholder(tf.float32, [ None, dim_data ])
    y_ = tf.placeholder(tf.float32, [ None, num_classes ])
    n_samples = tf.placeholder(tf.int32, [ 1 ])[ 0 ]
    kl_factor_ = tf.placeholder(tf.float32, [ 1 ])[ 0 ]
    training = tf.placeholder(tf.int32, shape = [ 1 ])[ 0 ]

    # Calculate the total number of parameters needed in the CNN

    total_weights_main = dict_dim_weights.copy()
    total_weights_main.update((x, np.prod(y)) for x, y in total_weights_main.items())

    total_weights = sum(total_weights_main.values())

    # NF
    # Create a new dictionary containing the samples from the mixing density

    dict_z = dict_dim_weights.copy()
    dict_z.update((x, y[0]) for x, y in dict_z.items())

    total_z = sum(dict_z.values())

    # Create the networks that compose the system

    generator = create_generator(n_units_gen, noise_comps_gen, total_z, n_layers_gen)
    discriminator = create_discriminator(n_units_disc, total_weights, n_layers_disc)
    main_CNN = create_main_NN(dict_dim_weights, dict_dim_biases)

    # Output the weights sampled from the generator

    z_samples = compute_output_generator(generator, n_samples, noise_comps_gen)

    # NF
    # From the z_samples and means and vars from main we need to sample the weights to use the discriminator

    weights, mean_weights, var_weights = sample_weights(main_CNN, z_samples, dict_dim_weights)

    # Obtain the moments of the weights and pass the values through the disc

    mean_weights , var_weights = tf.nn.moments(weights, axes = [0]) # The estimated means give wierd results!!!! This is better!

    mean_w = tf.stop_gradient(mean_weights)
    var_w = tf.stop_gradient(var_weights) + 1e-6

    # Normalize real weights

    norm_weights = (weights - mean_w) / tf.sqrt(var_w)

    # Generate samples of a normal distribution with the moments of the weights

    w_gaussian = tf.random_normal(shape = tf.shape(weights), mean = 0, stddev = 1, seed = seed)

    # Obtain the T(z,x) for the real and the sampled weights

    T_real = compute_output_discriminator(discriminator, norm_weights, n_layers_disc)
    T_sampled = compute_output_discriminator(discriminator, w_gaussian, n_layers_disc)

    # Calculate the cross entropy loss for the discriminator

    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=T_real, labels=tf.ones_like(T_real)))
    d_loss_sampled = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=T_sampled, labels=tf.zeros_like(T_sampled)))

    cross_entropy_per_point = (d_loss_real + d_loss_sampled) / 2.0

    # Obtain the KL and ELBO


#    logr = -0.5 * tf.reduce_sum(norm_weights**2 + tf.log(var_w) + np.log(2.0 * np.pi), [ 1 ])
#    logz = -0.5 * tf.reduce_sum((weights)**2 / tf.exp(log_vars_prior) + log_vars_prior + np.log(2.0 * np.pi), [ 1 ])
#    KL = (T_real + logr - logz)

#    KL = 0.0 * KL

    means_1 = main_CNN["mean_D1"]
    means_2 = main_CNN["mean_D2"]
    means_3 = main_CNN["mean_D3"]

    log_vars_1 = main_CNN["log_sigma2_D1"]
    log_vars_2 = main_CNN["log_sigma2_D2"]
    log_vars_3 = main_CNN["log_sigma2_D3"]


    log_vars_prior_1 = tf.Variable(tf.cast(np.log(1.0) * np.ones((200)), dtype = tf.float32))
    log_vars_prior_2 = tf.Variable(tf.cast(np.log(1.0) * np.ones((200)), dtype = tf.float32))
    log_vars_prior_3 = tf.Variable(tf.cast(np.log(1.0) * np.ones((10)), dtype = tf.float32))

    means_prior_1 = tf.Variable(tf.cast(0.0 * np.ones((200)), dtype = tf.float32))
    means_prior_2 = tf.Variable(tf.cast(0.0 * np.ones((200)), dtype = tf.float32))
    means_prior_3 = tf.Variable(tf.cast(0.0 * np.ones((10)), dtype = tf.float32))

    KL = tf.reduce_sum(0.5 * (means_1 - means_prior_1)**2 / tf.exp(log_vars_prior_1) + \
        0.5 * (tf.exp(log_vars_1 - log_vars_prior_1) - 1.0 - log_vars_1 + log_vars_prior_1))
    KL += tf.reduce_sum(0.5 * (means_2 - means_prior_2)**2 / tf.exp(log_vars_prior_2) + \
        0.5 * (tf.exp(log_vars_2 - log_vars_prior_2) - 1.0 - log_vars_2 + log_vars_prior_2))
    KL += tf.reduce_sum(0.5 * (means_3 - means_prior_3)**2 / tf.exp(log_vars_prior_3) + \
        0.5 * (tf.exp(log_vars_3 - log_vars_prior_3) - 1.0 - log_vars_3 + log_vars_prior_3))


    # ==========================================================================
    # SEPARATE WEIGHTS IN DICTIONARY TO USE THEM IN THE CNN
    # ==========================================================================

    res_train, log_prob_data, classification_error, alt_accuracy, forecasts, real_labels, multiclass_brier = \
        compute_outputs_main_NN(z_samples, main_CNN, dict_dim_weights, x, y_, n_samples, training)

    # Make the estimates of the ELBO for the primary classifier

    ELBO = (tf.reduce_sum(res_train) - kl_factor_ * tf.reduce_mean(KL) * tf.cast(tf.shape(x)[ 0 ], tf.float32) / \
        tf.cast(total_training_data, tf.float32)) * tf.cast(total_training_data, tf.float32) / tf.cast(tf.shape(x)[ 0 ], tf.float32)

    neg_ELBO = -ELBO
    main_loss = neg_ELBO
    mean_ELBO = ELBO

    # KL y res_train have shape batch_size x n_samples

    mean_KL = tf.reduce_mean(KL)

    # Create the variable lists to be updated

    vars_primal = get_variables_generator(generator) + get_variables_main_NN(main_CNN) #+ \
    vars_dual = get_variables_discriminator(discriminator)

    train_step_primal = tf.train.AdamOptimizer(primal_rate).minimize(main_loss, var_list = vars_primal)
    train_step_dual = tf.train.AdamOptimizer(dual_rate).minimize(cross_entropy_per_point, var_list = vars_dual)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.24)

    with tf.Session(config = tf.ConfigProto(gpu_options=gpu_options)) as sess:


        sess.run(tf.global_variables_initializer())

        total_ini = time.time()

        # Change the value of alpha to begin exploring using the second value given

        for epoch in range(n_epochs):

            L = 0.0
            ce_estimate = 0.0
            kl = 0.0
            class_error_estimate = 0.0
            mc_brier = 0.0
            ll_train = 0.0

            L_epoch = 0.0
            ce_estimate_epoch = 0.0
            kl_epoch = 0.0

            n_batches_train = int(np.ceil(total_training_data / n_batch))
            for i_batch in range(n_batches_train):

                kl_factor = np.minimum(epoch / (n_epochs * 0.2), 1.0)

                ini = time.clock()
                ini_ref = time.time()
                ini_train = time.clock()

                last_point = np.minimum(n_batch * (i_batch + 1), total_training_data)

                batch = [ X_train[ i_batch * n_batch : last_point, : ] , y_train[ i_batch * n_batch : last_point, : ] ]

                sess.run(train_step_primal, feed_dict={x: batch[ 0 ], y_: batch[ 1 ], n_samples: n_samples_train, \
                    kl_factor_: kl_factor, training : 0})

                Ltemp, kltemp, ce_estimate_temp, class_error_train, mc_brier_tmp, ll_tmp  = sess.run([ mean_ELBO, mean_KL, \
                    cross_entropy_per_point, classification_error, multiclass_brier, log_prob_data], feed_dict={x: batch[ 0 ], \
                    y_: batch[ 1 ], n_samples: n_samples_train, kl_factor_: kl_factor, training : 0})

                L += Ltemp
                kl += kltemp
                ce_estimate += ce_estimate_temp
                class_error_estimate += class_error_train
                mc_brier += mc_brier_tmp
                ll_train += ll_tmp

                L_epoch += Ltemp
                ce_estimate_epoch += ce_estimate_temp
                kl_epoch += kltemp

                fini = time.clock()
                fini_ref = time.time()
                fini_train = time.clock()

                sys.stdout.write('.')
                sys.stdout.flush()

                fini_train = time.clock()

                if ((i_batch + 1) % batches_to_report) == 0:
                    string = ('\n VI - batch %g datetime %s epoch %d ELBO %g CROSS-ENT %g KL %g annealing_factor %g \
                        real_time %g cpu_time %g train_time %g') % \
                        (i_batch + 1, str(datetime.now()), epoch + 1, \
                        L / batches_to_report, ce_estimate / batches_to_report, kl / batches_to_report, kl_factor, (fini_ref - \
                        ini_ref), (fini - ini), (fini_train - ini_train))
                    print(string)
                    sys.stdout.flush()
                    string_acc = ('Training error --- Error %g; MC-Brier score %g LL %g ') % (class_error_estimate / batches_to_report, mc_brier / batches_to_report, ll_train / batches_to_report)
                    print(string_acc)

                    L = 0.0
                    kl = 0.0
                    ce_estimate = 0.0
                    class_error_estimate = 0.0
                    mc_brier = 0.0
                    ll_train = 0.0

        # Test Evaluation

        sys.stdout.write('\n')
        ini_test = time.time()

        # We do the test evaluations
        cl_errors_cont = []
        LL_cont  = []
        acc_cont = []
        brier_cont = []
        n_batches_to_process = int(np.ceil(X_test.shape[ 0 ] / n_batch_test))

        for i in range(n_batches_to_process):

            last_point = np.minimum(n_batch_test * (i + 1), X_test.shape[ 0 ])

            batch = [ X_test[ i * n_batch_test : last_point, : ] , y_test[ i * n_batch_test : last_point, : ] ]

            LL_tmp, cl_error_tmp, acc_tmp, brier_tmp = sess.run([ log_prob_data, classification_error, alt_accuracy, \
            multiclass_brier ], feed_dict={x: batch[0], y_: batch[1], n_samples: n_samples_test, training : 0})

            cl_errors_cont.append( cl_error_tmp )
            LL_cont.append( LL_tmp )
            acc_cont.append( acc_tmp )
            brier_cont.append( brier_tmp )

        # import pdb; pdb.set_trace()

        error_class = np.mean(cl_errors_cont)
        TestLL = np.mean(LL_cont)
        accuracy_test = np.mean(acc_cont)
        brier_test = np.mean(brier_cont)

        error_class_var = np.std(cl_errors_cont) / len(cl_errors_cont)
        TestLL_var = np.std(LL_cont) / len(LL_cont)
        accuracy_test_var = np.std(acc_cont) / len(acc_cont)
        brier_test_var = np.std(brier_cont) / len(brier_cont)


        fini_test = time.time()
        fini = time.clock()
        fini_ref = time.time()
        total_fini = time.time()

        string = ('VI - batch %g datetime %s epoch %d ELBO %g CROSS-ENT %g KL %g real_time %g cpu_time %g ' + \
            'test_time %g total_time %g KL_factor %g ; LL %g - LL_var %g; Classification_error %g - Class_var %g ; ' + \
            'Accuracy %g - Acc_var %g ; Brier-score %g Brier_var %g') % \
            (i_batch, str(datetime.now()), epoch, \
            L_epoch / n_batches_train, ce_estimate_epoch / n_batches_train, kl_epoch / n_batches_train, (fini_ref - \
            ini_ref), (fini - ini), (fini_test - ini_test), (total_fini - total_ini), \
            kl_factor, TestLL, TestLL_var, error_class, error_class_var, accuracy_test, accuracy_test_var, brier_test, brier_test_var)
        print(string)
        sys.stdout.flush()

        final_forecasts, final_labels = sess.run([ forecasts, real_labels ], feed_dict={x: X_test[0:10], \
            y_: y_test[0:10], n_samples: n_samples_test, training : 0})

        np.savetxt('res_vi/results_error.txt', [ error_class ])
        np.savetxt('res_vi/results_ll.txt', [ TestLL ])
        np.savetxt('res_vi/results_accuracy.txt', [ accuracy_test ])
        np.savetxt('res_vi/results_brier.txt', [ brier_test ])

        np.savetxt('res_vi/results_error_var.txt', [ error_class_var ])
        np.savetxt('res_vi/results_ll_var.txt', [ TestLL_var ])
        np.savetxt('res_vi/results_accuracy_var.txt', [ accuracy_test_var ])
        np.savetxt('res_vi/results_brier_var.txt', [ brier_test_var ])

        np.savetxt('res_vi/forecasts.txt', final_forecasts)
        np.savetxt('res_vi/labels.txt', final_labels)




if __name__ == '__main__':

    # Create the folder to save all the results
    if not os.path.isdir("res_vi"):
        os.makedirs("res_vi")

    main( )

