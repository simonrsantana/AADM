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

seed = 123

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

n_samples_train = 10
n_samples_test = 100

n_batch = 100
n_epochs = 50 # We do not expect the algorithm to fulfill more than a few epochs before it reaches convergence
n_size_test = 10000
n_batches_to_report = 100

# Structural parameters of the main NN

n_units_nn = 50
n_layers_nn = 2  # Number of layers in the NN

n_layers_gen = 2          # Number of layers in the generator
n_units_gen = 50          # Number of units in the generator for 1 hidden layer in the VAE
noise_comps_gen = 100     # Number of gaussian variables inputed in the encoder

# Parameters for the discriminative network

n_units_disc = 50
n_layers_disc = 2

# Learning rates

primal_rate = 1e-5 # Actual Bayesian NN
dual_rate = 1e-4   # Discriminator 

# We define the following two functions to simplify the rest of the code

def w_variable_mean(shape):
  initial = tf.random_normal(shape = shape, stddev = 0.1, seed = seed) # mean 0 stddev 1
  return tf.Variable(initial)

def w_variable_variance(shape):
  initial = tf.random_normal(shape = shape, stddev = 0.1, seed = seed) - 5.0 # mean 0 stddev 1
  return tf.Variable(initial)


###############################################################################
##########################   Network structure  ###############################
##############################################################################a

def create_generator(n_units_gen, noise_comps_gen, total_number_weights, n_layers_gen):

    mean_noise = w_variable_mean([ 1, 1, noise_comps_gen ])
    log_var_noise = w_variable_variance([ 1, 1, noise_comps_gen ])

    W1_gen = w_variable_mean([ noise_comps_gen, n_units_gen ])
    bias1_gen  = w_variable_mean([ n_units_gen ])

    W2_gen = w_variable_mean([ n_units_gen, n_units_gen ])
    bias2_gen  = w_variable_mean([ n_units_gen ])

    W3_gen = w_variable_mean([ n_units_gen, total_number_weights ])
    bias3_gen  = w_variable_mean([ total_number_weights ])

    return {'mean_noise': mean_noise, 'log_var_noise': log_var_noise, 'W1_gen': W1_gen, 'bias1_gen': bias1_gen, \
        'W2_gen': W2_gen, 'bias2_gen': bias2_gen, 'W3_gen': W3_gen, 'bias3_gen': bias3_gen, 'n_layers_gen': n_layers_gen}

def get_variables_generator(generator):
    return [ generator['mean_noise'], generator['log_var_noise'], generator['W1_gen'], \
        generator['W2_gen'], generator['W3_gen'], generator['bias1_gen'], generator['bias2_gen'], generator['bias3_gen'] ]

def compute_output_generator(generator, batchsize, n_samples, noise_comps_gen):

    mean_noise = generator['mean_noise']
    log_var_noise = generator['log_var_noise']
    W1_gen = generator['W1_gen']
    W2_gen = generator['W2_gen']
    W3_gen = generator['W3_gen']

    bias1_gen = generator['bias1_gen']
    bias2_gen = generator['bias2_gen']
    bias3_gen = generator['bias3_gen']

    pre_init_noise = tf.random_normal(shape = [ batchsize, n_samples, noise_comps_gen ], seed = seed)

    init_noise =  mean_noise + tf.sqrt(tf.exp( log_var_noise )) * pre_init_noise

    # Process the noises through the network

    A1_gen = tf.tensordot(init_noise, W1_gen, axes = [[2], [0]]) + bias1_gen
    h1_gen = tf.nn.leaky_relu(A1_gen)

    if generator['n_layers_gen'] == 1:
        A3_gen = tf.tensordot(h1_gen, W3_gen, axes = [[2], [0]]) + bias3_gen    # final weights
    else:
        A2_gen = tf.tensordot(h1_gen, W2_gen, axes = [[2], [0]]) + bias2_gen
        h2_gen = tf.nn.leaky_relu(A2_gen)
        A3_gen = tf.tensordot(h2_gen, W3_gen, axes = [[2], [0]]) + bias3_gen    # final weights

    return A3_gen

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

    A1_disc = tf.tensordot(weights, W1_disc, axes = [[2], [0]]) + bias1_disc
    h1_disc = tf.nn.leaky_relu(A1_disc)

    if discriminator['n_layers_disc'] == 2:
        A2_disc = tf.tensordot(h1_disc, W2_disc, axes = [[2], [0]]) + bias2_disc
        h2_disc = tf.nn.leaky_relu(A2_disc)

        A3_disc = tf.tensordot(h2_disc, W3_disc, axes = [[2], [0]]) + bias3_disc
    else:
        A3_disc = tf.tensordot(h1_disc, W3_disc, axes = [[2], [0]]) + bias3_disc

    return A3_disc[ :, :, 0 ]

def create_main_NN(n_units, n_layers):

    # Only the variables of that Network (prior variance and noise variance)

    log_vars_prior = tf.Variable(tf.cast(np.log(1.0), dtype = tf.float32))
    log_sigma2_noise = tf.Variable(tf.cast((1.0 / 10.0), dtype = tf.float32))

    # The biases are not random

    bias_A1 = w_variable_mean([ n_units ])
    bias_A2 = w_variable_mean([ n_units ])
    bias_A3 = w_variable_mean([ 1 ])

    return {'log_vars_prior': log_vars_prior, 'log_sigma2_noise': log_sigma2_noise, \
        'n_units': n_units, 'bias_A1': bias_A1, 'bias_A2': bias_A2, 'bias_A3': bias_A3, 'n_layers': n_layers}

def get_variables_main_NN(network):
    return [ network['log_sigma2_noise'], network['log_vars_prior'], network['bias_A1'], network['bias_A2'], network['bias_A3'] ]

def compute_outputs_main_NN(network, x_input, y_target, mean_targets, std_targets, weights, alpha, n_samples, dim_data):

    log_vars_prior = network['log_vars_prior']
    log_sigma2_noise = network['log_sigma2_noise']
    n_units = network['n_units']
    bias_A1 = network['bias_A1']
    bias_A2 = network['bias_A2']
    bias_A3 = network['bias_A3']

    batch_size = tf.shape(x_input)[ 0 ]
    W1 = tf.reshape(weights[:,:, : (dim_data * n_units) ], shape = [ batch_size, n_samples, n_units, dim_data ])

    if network['n_layers'] == 2:
        W2 = tf.reshape(weights[:,:, (dim_data * n_units) : (dim_data * n_units + n_units * n_units) ], \
        shape = [ batch_size, n_samples, n_units, n_units ])
    else:
        W2 = tf.reshape(weights[:,:, (dim_data * n_units) : ], shape = [ batch_size, n_samples, n_units, 1 ])

    W3 = tf.reshape(weights[:,:, (dim_data * n_units + n_units * n_units) : ], shape = [ batch_size, n_samples, n_units, 1 ])

    A1 = tf.reduce_sum(tf.reshape(x_input, shape = [ batch_size, 1, 1, dim_data]) * W1, axis = 3) + bias_A1
    h1 = tf.nn.leaky_relu(A1) # h1 is batch_size x n_samples x n_units

    if network['n_layers'] == 2:
        A2 = tf.reduce_sum(tf.reshape(h1, shape = [ batch_size, n_samples, n_units, 1 ]) * W2, axis = 2) + bias_A2
        h2 = tf.nn.leaky_relu(A2) # h2 is batch_size x n_samples x n_units
        A3 = tf.reduce_sum(tf.reshape(h2, shape = [ batch_size, n_samples, n_units, 1 ]) * W3, axis = 2) + bias_A3
    else:
        A3 = tf.reduce_sum(tf.reshape(h1, shape = [ batch_size, n_samples, n_units, 1 ]) * W2, axis = 2) + bias_A3

    res_train = 1.0 / alpha * (- tf.log(tf.cast(n_samples, tf.float32)) + tf.reduce_logsumexp( alpha * (-0.5 * \
        (np.log(2.0 * np.pi) + log_sigma2_noise + (A3 - tf.reshape(y_target, shape = [ tf.shape(x_input)[0], 1, 1 ]))**2 \
        / tf.exp(log_sigma2_noise))) , axis = [ 1 ]))

    # This is to compute the test log loglikelihood

    log_prob_data = tf.reduce_sum(((tf.reduce_logsumexp(-0.5 * tf.log(2.0 * np.pi * (tf.exp(log_sigma2_noise) * std_targets**2)) \
        - 0.5 * (A3 * std_targets + mean_targets - tf.reshape(y_target, shape = \
        [ tf.shape(x_input)[0], 1, 1 ]))**2 / (tf.exp(log_sigma2_noise) * std_targets**2), axis = [ 1 ])) \
        - tf.log(tf.cast(n_samples, tf.float32))))

    error = tf.reduce_sum((tf.reduce_mean(A3, axis = [ 1 ]) * std_targets + mean_targets - y_target)**2)

    return res_train, error, log_prob_data

###############################################################################
###############################################################################
###############################################################################

def main(alpha, layers):

    np.random.seed(seed)
    tf.set_random_seed(seed)

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

    # Create the model

    dim_data = X_train.shape[ 1 ]

    # Placeholders for data and number of samples

    x = tf.placeholder(tf.float32, [ None, dim_data ])
    y_ = tf.placeholder(tf.float32, [ None, 1 ])
    n_samples = tf.placeholder(tf.int32, [ 1 ])[ 0 ]

    n_layers_nn = n_layers_gen = n_layers_disc = layers

    if n_layers_nn == 2:
        total_weights = n_units_nn * (dim_data + n_units_nn) + n_units_nn # Number of weights for the 2 hidden layers case
    else: 
        total_weights = (dim_data + 1) * n_units_nn  # Total number of weights used

    generator = create_generator(n_units_gen, noise_comps_gen, total_weights, n_layers_gen)
    discriminator = create_discriminator(n_units_disc, total_weights, n_layers_disc)
    main_NN = create_main_NN(n_units_nn, n_layers_nn)

    weights = compute_output_generator(generator, tf.shape(x)[ 0 ], n_samples, noise_comps_gen)

    # Obtain the moments of the weights and pass the values through the disc

    mean_w , var_w = tf.nn.moments(weights, axes = [0, 1])

    mean_w = tf.stop_gradient(mean_w)
    var_w = tf.stop_gradient(var_w)

    # Normalize real weights

    norm_weights = (weights - mean_w) / tf.sqrt(var_w)

    # Generate samples of a normal distribution with the moments of the weights

    w_gaussian = tf.random_normal(shape = tf.shape(weights), mean = 0, stddev = 1, seed = seed)

    # Obtain the T(z,x) for the real and the sampled weights

    T_real = compute_output_discriminator(discriminator, norm_weights, layers)
    T_sampled = compute_output_discriminator(discriminator, w_gaussian, layers)

    # Calculate the cross entropy loss for the discriminator

    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=T_real, labels=tf.ones_like(T_real)))
    d_loss_sampled = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=T_sampled, labels=tf.zeros_like(T_sampled)))

    cross_entropy_per_point = (d_loss_real + d_loss_sampled) / 2.0

    # Obtain the KL and ELBO

    logr = -0.5 * tf.reduce_sum(norm_weights**2 + tf.log(var_w) + np.log(2.0 * np.pi), [ 2 ])
    logz = -0.5 * tf.reduce_sum((weights)**2 / tf.exp(main_NN['log_vars_prior']) + main_NN['log_vars_prior'] + np.log(2.0 * np.pi), [ 2 ])
    KL = T_real + logr - logz

    res_train, error, log_prob_data = compute_outputs_main_NN(main_NN, x, y_, mean_targets, std_targets, weights, \
        alpha, n_samples, dim_data)
 
    # Make the estimates of the ELBO for the primary classifier

    ELBO = (tf.reduce_sum(res_train) - tf.reduce_mean(KL) * tf.cast(tf.shape(x)[ 0 ], tf.float32) / \
        tf.cast(total_training_data, tf.float32)) * tf.cast(total_training_data, tf.float32) / tf.cast(tf.shape(x)[ 0 ], tf.float32)

    neg_ELBO = -ELBO
    main_loss = neg_ELBO
    mean_ELBO = ELBO

    # KL y res_train have shape batch_size x n_samples

    mean_KL = tf.reduce_mean(KL)

    # Create the variable lists to be updated

    vars_primal = get_variables_generator(generator) + get_variables_main_NN(main_NN)
    vars_dual = get_variables_discriminator(discriminator)

    train_step_primal = tf.train.AdamOptimizer(primal_rate).minimize(main_loss, var_list = vars_primal)
    train_step_dual = tf.train.AdamOptimizer(dual_rate).minimize(cross_entropy_per_point, var_list = vars_dual)

    config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1, \
        allow_soft_placement=True, device_count = {'CPU': 1})

    with tf.Session(config = config) as sess:

        sess.run(tf.global_variables_initializer())

        total_ini = time.time()

        # Change the value of alpha to begin exploring using the second value given

        for epoch in range(n_epochs):

            L = 0.0
            ce_estimate = 0.0
            kl = 0.0

            for i_batch in range(int(np.ceil(size_train / n_batch))):

                ini = time.clock()
                ini_ref = time.time()
                ini_train = time.clock()

                last_point = np.minimum(n_batch * (i_batch + 1), size_train)

                batch = [ X_train[ i_batch * n_batch : last_point, : ] , y_train[ i_batch * n_batch : last_point, ] ]
                
                sess.run(train_step_dual, feed_dict={x: batch[ 0 ], y_: batch[ 1 ], n_samples: n_samples_train})
                sess.run(train_step_primal, feed_dict={x: batch[ 0 ], y_: batch[ 1 ], n_samples: n_samples_train})

                L += sess.run(mean_ELBO, feed_dict={x: batch[ 0 ], y_: batch[ 1 ], n_samples: n_samples_train})
                kl += sess.run(mean_KL, feed_dict={x: batch[ 0 ], y_: batch[ 1 ], n_samples: n_samples_train})
                ce_estimate += sess.run(cross_entropy_per_point, feed_dict={x: batch[ 0 ], y_: batch[ 1 ], n_samples: n_samples_train})

#                sys.stdout.write('.')
#                sys.stdout.flush()

                fini_train = time.clock()

                if i_batch % n_batches_to_report == 0 and not (i_batch == 0 and epoch == 0):

                    sys.stdout.write('\n')
                    ini_test = time.time()

                    # We do the test evaluation RMSE

                    SE = 0.0
                    LL  = 0.0
                    for i in range(int(np.ceil(X_test.shape[ 0 ] / n_batch))):

                        last_point = np.minimum(n_batch * (i + 1), X_test.shape[ 0 ])

                        batch = [ X_test[ i * n_batch : last_point, : ] , y_test[ i * n_batch : last_point, ] ]

                        SE += sess.run(error, feed_dict={x: batch[0], y_: batch[1], n_samples: n_samples_test})
                        LL += sess.run(log_prob_data, feed_dict={x: batch[ 0 ], y_: batch[ 1 ], n_samples: n_samples_test})

                    RMSE = np.sqrt(SE / float(X_test.shape[ 0 ]))
                    TestLL = (LL / float(X_test.shape[ 0 ]))

                    fini_test = time.time()
                    fini = time.clock()
                    fini_ref = time.time()
                    total_fini = time.time()
        
                    with open("results_AADM_airlines/res_alpha_" + str(alpha) + "_airlines_350.txt", "a") as res_file:
                        string = ('alpha %g batch %g datetime %s epoch %d ELBO %g CROSS-ENT %g KL %g real_time %g cpu_time %g ' + \
                            'train_time %g test_time %g total_time %g LL %g RMSE %g') % (alpha, i_batch, str(datetime.now()), epoch, \
                            L / n_batches_to_report, ce_estimate / n_batches_to_report, kl / n_batches_to_report, (fini_ref - \
                            ini_ref), (fini - ini), (fini_train - ini_train), (fini_test - ini_test), (total_fini - total_ini), TestLL, \
                            RMSE)
                        res_file.write(string + "\n")
                        print(string)
                        sys.stdout.flush()
            
                    L = 0.0
                    ce_estimate = 0.0
                    kl = 0.0

if __name__ == '__main__':


    if not os.path.isdir("results_AADM_airlines"):
        os.makedirs("results_AADM_airlines")

    alpha = np.float(sys.argv[1])
    layers = int(sys.argv[2])

    if os.path.isfile("results_AADM_airlines/res_alpha_" + str(alpha) + "_airlines.txt"):
        with open("results_AADM_airlines/res_alpha_" + str(alpha) + "_airlines.txt", "w") as file:
            file.close()


    main(alpha, layers)
