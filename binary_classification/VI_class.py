################################################
##### VARIATIONAL INFERENCE for regression #####
################################################

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
# from properscoring import brier_score
from _brier import brier_score

import os
os.chdir(".")

seed = 123

# =============================================================================
# Complete system parameters (VAE, encoder, decoder, data specs, etc.)
# =============================================================================

# File to be analyzed

original_file = sys.argv[ 3 ]

# This is the total number of training samples

n_samples_train = 10
n_samples_test = 100

n_batch = 10
n_epochs = 4000

# Structural parameters of the main NN

n_units_nn = 50
n_layers_nn = 2  # Number of layers in the NN

# Learning rates

primal_rate = 1e-4 # Actual Bayesian NN

# This is the total number of training samples

kl_factor_limit = int(n_epochs / 10)

ratio_train = 0.9 # Percentage of the data devoted to train

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

def create_generator(total_number_weights):

    mean_noise = w_variable_mean([ 1, 1, total_number_weights ])
    log_var_noise = w_variable_variance([ 1, 1, total_number_weights ])

    return {'mean_noise': mean_noise, 'log_var_noise': log_var_noise }

def get_variables_generator(generator):
    return [ generator['mean_noise'], generator['log_var_noise'] ]

def compute_output_generator(generator, batchsize, n_samples, total_number_weights):

    mean_noise = generator['mean_noise']
    log_var_noise = generator['log_var_noise']

    pre_init_noise = tf.random_normal(shape = [ batchsize, n_samples, total_number_weights ], seed = seed)

    init_noise =  mean_noise + tf.sqrt(tf.exp( log_var_noise )) * pre_init_noise

    return init_noise

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

def compute_outputs_main_NN(network, x_input, y_target, weights, n_samples, dim_data):

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

    res_train = tf.reduce_mean(-1.0 * tf.nn.sigmoid_cross_entropy_with_logits(labels = y_target[:,None,], logits = A3), axis = [ 1 ])

    # This is to compute the test log loglikelihood

    log_prob_data = tf.reduce_sum(((tf.reduce_logsumexp(-1.0 * tf.nn.sigmoid_cross_entropy_with_logits(\
        labels = y_target[:,None,], logits = A3), axis = [ 1 ])) - tf.log(tf.cast(n_samples, tf.float32))))

    error = tf.reduce_sum(tf.math.abs(tf.math.round(tf.reduce_mean(tf.math.sigmoid(A3), axis = [ 1 ])) - y_target))

    y_test = tf.math.sigmoid(A3)[:,:,0]

    return res_train, error, log_prob_data, y_test

###############################################################################
###############################################################################
###############################################################################

def main(permutation, split, layers):

    np.random.seed(seed)
    tf.set_random_seed(seed)

    # We load the original dataset

    data = np.loadtxt(original_file)

    # =========================================================================
    #  Parameters of the complete system
    # =========================================================================

    # We obtain the features and the targets

    X = data[ :, range(data.shape[ 1 ] - 1) ]
    y = data[ :, data.shape[ 1 ] - 1 ]

    # We create the train and test sets with 100*ratio % of the data

    data_size = X.shape[ 0 ]
    size_train = int(np.round(data_size * ratio_train))
    total_training_data = size_train

    index_train = permutation[ 0 : size_train ]
    index_test = permutation[ size_train : ]

    X_train = X[ index_train, : ]
    y_train = np.vstack(y[ index_train ])
    X_test = X[ index_test, : ]
    y_test = np.vstack(y[ index_test ])

    # Normalizamos los argumentos

    meanXTrain = np.mean(X_train, axis = 0)
    stdXTrain = np.std(X_train, axis = 0)

    meanyTrain = np.mean(y_train)
    stdyTrain = np.std(y_train)

    X_train = (X_train - meanXTrain) / stdXTrain
    X_test = (X_test - meanXTrain) / stdXTrain

    # Create the model

    dim_data = X_train.shape[ 1 ]

    # Placeholders for data and number of samples

    x = tf.placeholder(tf.float32, [ None, dim_data ])
    y_ = tf.placeholder(tf.float32, [ None, 1 ])
    n_samples = tf.placeholder(tf.int32, [ 1 ])[ 0 ]
    kl_factor_ = tf.placeholder(tf.float32, [ 1 ])[ 0 ]

    n_layers_nn = n_layers_gen = n_layers_disc = layers

    if n_layers_nn == 2:
        total_weights = n_units_nn * (dim_data + n_units_nn) + n_units_nn # Number of weights for the 2 hidden layers case
    else:
        total_weights = (dim_data + 1) * n_units_nn  # Total number of weights used

    generator = create_generator(total_weights)
    main_NN = create_main_NN(n_units_nn, n_layers_nn)

    weights = compute_output_generator(generator, tf.shape(x)[ 0 ], n_samples, total_weights)

    # Obtain the KL and ELBO

    KL = 0.5 * tf.reduce_sum(tf.exp(generator[ 'log_var_noise' ] - main_NN['log_vars_prior']) + \
        (0.0 - generator[ 'mean_noise' ])**2 / tf.exp(main_NN['log_vars_prior']) - 1.0 + main_NN['log_vars_prior'] \
        - generator[ 'log_var_noise' ])

    res_train, error, log_prob_data, forecasted_classes = compute_outputs_main_NN(main_NN, x, y_, weights, \
        n_samples, dim_data)

    # Make the estimates of the ELBO for the primary classifier

    ELBO = (tf.reduce_sum(res_train) - kl_factor_ * tf.reduce_mean(KL) * tf.cast(tf.shape(x)[ 0 ], tf.float32) / \
        tf.cast(total_training_data, tf.float32)) * tf.cast(total_training_data, tf.float32) / tf.cast(tf.shape(x)[ 0 ], tf.float32)

    neg_ELBO = -ELBO
    main_loss = neg_ELBO
    mean_ELBO = ELBO

    # KL y res_train have shape batch_size x n_samples

    mean_KL = tf.reduce_mean(KL)

    # Create the variable lists to be updated

    vars_primal = get_variables_generator(generator) + get_variables_main_NN(main_NN)

    train_step_primal = tf.train.AdamOptimizer(primal_rate).minimize(main_loss, var_list = vars_primal)

    config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1, \
        allow_soft_placement=True, device_count = {'CPU': 1})

    with tf.Session(config = config) as sess:

        sess.run(tf.global_variables_initializer())

        total_ini = time.time()

        # Change the value of to begin exploring using the second value given

        for epoch in range(n_epochs):

            L = 0.0
            ce_estimate = 0.0
            kl = 0.0

            kl_factor = np.minimum(1.0 * epoch / kl_factor_limit, 1.0)

            n_batches_train = int(np.ceil(size_train / n_batch))
            for i_batch in range(n_batches_train):

                ini = time.clock()
                ini_ref = time.time()
                ini_train = time.clock()

                last_point = np.minimum(n_batch * (i_batch + 1), size_train)

                batch = [ X_train[ i_batch * n_batch : last_point, : ] , y_train[ i_batch * n_batch : last_point, ] ]

                sess.run(train_step_primal, feed_dict={x: batch[ 0 ], y_: batch[ 1 ], n_samples: n_samples_train, \
                    kl_factor_: kl_factor})

                L += sess.run(mean_ELBO, feed_dict={x: batch[ 0 ], y_: batch[ 1 ], n_samples: n_samples_train, \
                    kl_factor_: kl_factor})
                kl += sess.run(mean_KL, feed_dict={x: batch[ 0 ], y_: batch[ 1 ], n_samples: n_samples_train})

                sys.stdout.write('.')
                sys.stdout.flush()

                fini_train = time.clock()
                fini_ref = time.time()


            string = ('VI - batch %g datetime %s epoch %d ELBO %g CROSS-ENT %g KL %g cpu_time %g ' + \
                'train_time %g KL_factor %g') % \
                (i_batch, str(datetime.now()), epoch, \
                L / n_batches_train, ce_estimate / n_batches_train, kl / n_batches_train, (fini_ref - \
                ini_ref), (fini_train - ini_train), kl_factor)
            print(string)
            sys.stdout.flush()

        # Test Evaluation

        sys.stdout.write('\n')
        ini_test = time.time()

        # We do the test evaluation RMSE

        errors = 0.0
        LL  = 0.0
        n_batches_to_process = int(np.ceil(X_test.shape[ 0 ] / n_batch))
        for i in range(n_batches_to_process):

            last_point = np.minimum(n_batch * (i + 1), X_test.shape[ 0 ])

            batch = [ X_test[ i * n_batch : last_point, : ] , y_test[ i * n_batch : last_point, ] ]

            errors_tmp, LL_tmp = sess.run([error, log_prob_data], feed_dict={x: batch[0], y_: batch[1], n_samples: n_samples_test})
            errors += errors_tmp
            LL += LL_tmp

        error_class = errors / float(X_test.shape[ 0 ])
        TestLL = (LL / float(X_test.shape[ 0 ]))

        # Estimate the Brier score for the binary classification
        results, labels = sess.run([forecasted_classes, y_], feed_dict={x: X_test, y_: y_test, n_samples: n_samples_test})

        raw_brier_results = brier_score(labels.T, np.mean(results, axis = 1))
        mean_brier_results = np.mean(raw_brier_results)

        fini_test = time.time()
        fini = time.clock()
        total_fini = time.time()

        string = ('VI - batch %g datetime %s epoch %d ELBO %g CROSS-ENT %g KL %g real_time %g cpu_time %g ' + \
            'train_time %g test_time %g total_time %g KL_factor %g LL %g Error %g') % \
            (i_batch, str(datetime.now()), epoch, \
            L / n_batches_train, ce_estimate / n_batches_train, kl / n_batches_train, (fini_ref - \
            ini_ref), (fini - ini), (fini_train - ini_train), (fini_test - ini_test), (total_fini - total_ini), \
            kl_factor, TestLL, error_class)
        print(string)
        sys.stdout.flush()

        np.savetxt('res_vi/results_error_' + str(split) + '.txt', [ error_class ])
        np.savetxt('res_vi/results_ll_' + str(split) + '.txt', [ TestLL ])
        np.savetxt('res_vi/results_raw_Brier_' + str(split) + '.txt', raw_brier_results)
        np.savetxt('res_vi/results_mean_Brier_' + str(split) + '.txt', [ mean_brier_results ])



if __name__ == '__main__':

    split = int(sys.argv[1])
    layers = int(sys.argv[2])

    available_perm = np.loadtxt('permutations_' + original_file, delimiter = ",", dtype = int)

    # Create the storage folder
    if not os.path.isdir("res_vi"):
        os.makedirs("res_vi")


    main(available_perm[split,], split, layers)


