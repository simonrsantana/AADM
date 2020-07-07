###############################################################################
############### ADVERSARIAL VARIATIONAL BAYES in MNIST with VI ################
###############################################################################
#
#   Now we have to sample the weights instead of the activations in order to be
# able to extract them from an underlying network, using the Adaptive Contrast
# technique introduced in the AVB paper
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

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)

import os
os.chdir(".")

seed = 123

# =============================================================================
# DATA PARAMETERS AND SYSTEM ARCHITECTURE
# =============================================================================

batches_to_report = 50
# File to be analyzed
original_file = "mnist"

# This is the total number of training samples
n_batch = 50
n_batch_test = 1    # Since we will need more samples in test, we make the batchsize smaller to keep everything in manageable sizes
num_classes = 10
n_epochs = 50

# input image dimensions
img_width, img_height = 28, 28
channels_img = 1

n_samples_train = 10
n_samples_test = 100

# Structural parameters of the NN system
n_layers_gen = 2          # Number of layers in the generator
n_units_gen = 200          # Number of units in the generator for 1 hidden layer in the VAE
noise_comps_gen = 100     # Number of gaussian variables inputed in the encoder

# Parameters for the discriminative network
n_units_disc = 200
n_layers_disc = 2

# Learning rates
primal_rate = 1e-4 # Actual Bayesian NN
# dual_rate = 1e-3   # Discriminator

# ==============================================================================
# DEFINE HERE THE STRUCTURE OF THE CNN
# ==============================================================================

padding = "VALID" #or "SAME"
n_filters_1 = 32
n_filters_2 = 64
n_units_dense_1 = 128
n_classes = 10

dropout_rate = 0.25     # Probability of units turned off

fh1 = 5; fw1 = 5    # First filter
fh2 = 3; fw2 = 3    # Second filter

dict_weights = {
    'c1': [fh1, fw1, 1, n_filters_1], 'c2': [fh2, fw2, n_filters_1, n_filters_2],   # Convolution Layers
    'd1': [5*5*n_filters_2,n_units_dense_1], 'out': [n_units_dense_1,n_classes]     # Dense Layers
}

dict_biases = {
    'c1': n_filters_1, 'c2': n_filters_2,       # Convolution Layers
    'd1': n_units_dense_1, 'out': n_classes,    # Dense Layers
}

# We define the following two functions to simplify the rest of the code

def w_variable_mean(shape):
  initial = tf.random_normal(shape = shape, stddev = 0.1, seed = seed) # mean 0 stddev 1
  return tf.Variable(initial)

def w_variable_variance(shape):
  initial = tf.random_normal(shape = shape, stddev = 0.1, seed = seed) - 5.0 # mean 0 stddev 1
  return tf.Variable(initial)

def one_hot(a, num_classes):
  return np.squeeze(np.eye(num_classes)[a.reshape(-1)])


###############################################################################
##########################   Network structure  ###############################
###############################################################################

# Generator (encoder)

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


# ==============================================================================
# CONVOLUTIONAL NN MAIN
# ==============================================================================

def create_main_CNN(dict_biases):

    # The biases are not random

    bias_C1 = w_variable_mean([ dict_biases["c1"] ])
    bias_C2 = w_variable_mean([ dict_biases["c2"] ])
    bias_D1 = w_variable_mean([ dict_biases["d1"] ])
    bias_OUT = w_variable_mean([ dict_biases["out"] ])

    log_vars_prior = tf.Variable(tf.cast(np.log(1.0), dtype = tf.float32))

    return {'bias_C1': bias_C1, 'bias_C2': bias_C2, 'bias_D1': bias_D1, 'bias_OUT': bias_OUT, 'log_vars_prior': log_vars_prior}

def get_variables_main_CNN(network):
    return [ network['bias_C1'], network['bias_C2'], network['bias_D1'], network['bias_OUT'], network['log_vars_prior'] ]


# All the variables have been created before, now we only compute the ouptuts

# Main CNN function
#-------------------
# Inputs:
#   dict_total_weights  :           Dictionary with all the weights needed in each layer (without biases)
#   dict_weights        :           Dictionary containing the weights shape for each layer (excluding sample and batch sizes)
#   dict_biases         :           Dictionary containing the biases shape for each layer (excluding sample and batch sizes)
#   x_input             :           Batch of input data (train or test), shape (batchsize, img_height, img_width, channels_img)
#   y_target            :           One-hot encoded vector of the data class, shape (batchsize, n_classes)
#   weights             :           Tensor containing all the sampled weights and biases from the generator, shape (batchsize, n_samples, total_number_weights)
#   biases              :           Biases created in "create_main_CNN"
#   n_samples           :           Number of samples to be employed
#   training            :           Pseudo-boolean (1 if training, 0 while testing)
#
# Outputs:
#
#

def compute_outputs_main_CNN(dict_total_weights, dict_weights, dict_biases, x_input, y_target, weights, biases, n_samples, training = 0):

    # We need to tile x_input to apply one different sample of each filter to each input

    batch_size = tf.shape(x_input)[ 0 ]

    # weights = tf.tile(weights, [batch_size, n_samples, 1])
    tiled_x = tf.reshape(tf.tile(tf.expand_dims(x_input, 0), [n_samples, 1, 1, 1, 1]), [batch_size * n_samples, img_height, img_width, channels_img] )

    # Extract the biases from the generated network

    bias_C1 = biases["bias_C1"]
    bias_C2 = biases["bias_C2"]
    bias_D1 = biases["bias_D1"]
    bias_OUT = biases["bias_OUT"]

    # Cut the weights tensor in pieces needed for the layers here

    splits = [dict_total_weights["c1"], dict_total_weights["c2"], dict_total_weights["d1"], dict_total_weights["out"]]
    pre_W_C1 = weights[:,:, : splits[0]]
    pre_W_C2 = weights[:,:, splits[0] : (splits[0] + splits[1])]
    pre_W_D1 = weights[:,:, (splits[0] + splits[1]) : (splits[0] + splits[1] + splits[2])]
    pre_W_OUT = weights[:,:,(splits[0] + splits[1] + splits[2]) : ]

    # Reshape the weights tensors so they can be used in depthwise_conv2d
    filter_1 = dict_weights["c1"]
    filter_2 = dict_weights["c2"]

    # Filters have to have shape = (filter_height, filter_width, batch_size*n_samples, channels_in, channels_out)

    W_C1 = tf.reshape(pre_W_C1, shape = [batch_size * n_samples, filter_1[0], filter_1[1], filter_1[2], filter_1[3]])
    W_C1 = tf.transpose(W_C1, [ 1, 2, 0, 3, 4])
    W_C2 = tf.reshape(pre_W_C2, shape = [batch_size * n_samples, filter_2[0], filter_2[1], filter_2[2], filter_2[3]])
    W_C2 = tf.transpose(W_C2, [ 1, 2, 0, 3, 4])


    #############################
    # First convolutional layer #
    #############################

    tiled_x = tf.transpose(tiled_x, [1, 2, 0, 3]) # shape (height, width, batchsize * nsamples, channels_img)
    tiled_x = tf.reshape(tiled_x, [1, img_height, img_width, batch_size*n_samples*channels_img])

    # Reshape the filter so it fits the depthwise_conv2d function
    W_C1 = tf.reshape(W_C1, [filter_1[0], filter_1[1], filter_1[2] * batch_size * n_samples, filter_1[3]])

    A1 = tf.nn.depthwise_conv2d(tiled_x, filter = W_C1, strides=[1, 1, 1, 1], padding=padding)     # Filter should be (batchsize*nsamples, fh, fw, n_filters_1)

   # Now out shape is (1, img_height-fh+1, img_width-fw+1, batchsize * n_samples * channels_img * n_filters_1), because we used "VALID"

    # Dimensions of the output if padding = "VALID"
    conv_height_1 = int(img_height - filter_1[0] + 1)
    conv_width_1 = int(img_width - filter_1[1] + 1)

    if padding == "SAME":
        A1 = tf.reshape(A1, [img_height, img_width, batch_size * n_samples, channels_img, n_filters_1])
    if padding == "VALID":
        A1 = tf.reshape(A1, [conv_height_1, conv_width_1, batch_size * n_samples, channels_img, n_filters_1])


    A1 = tf.transpose(A1, [2, 0, 1, 3, 4])
    A1 = tf.reduce_sum(A1, axis=3)

    # Final activations for the first convolutional layer
    A1 = tf.nn.bias_add(A1, bias_C1)
    h1 = tf.nn.relu(A1)

    #---------#
    # POOLING #
    #---------#
    h1 = tf.nn.max_pool(h1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = padding)

    ##############################
    # Second convolutional layer #
    ##############################

    # Reshape the tensors so they fit the depthwise_conv2d function
    h1 = tf.transpose(h1, [1, 2, 0, 3])
    if padding == "SAME":
        h1 = tf.reshape(h1, [1, int(img_height / 2), int(img_width / 2), batch_size*n_samples*n_filters_1])
    if padding == "VALID":
        h1 = tf.reshape(h1, [1, int(conv_height_1 / 2), int(conv_width_1 / 2), batch_size*n_samples*n_filters_1])

    W_C2 = tf.reshape(W_C2, [filter_2[0], filter_2[1], filter_2[2] * batch_size * n_samples, filter_2[3]])

    A2 = tf.nn.depthwise_conv2d( h1, filter = W_C2, strides=[1, 1, 1, 1], padding=padding)     # Filter should be (batchsize*nsamples, fh, fw, n_filters_1)
    # Now out shape is (1, h1_h-fh+1, h1_w-fw+1, batchsize * n_samples * n_filters_1 * n_filters_2), because we used "VALID"

    # Dimensions of the output if padding = "VALID"
    conv_height_2 = int(conv_height_1 / 2 - filter_2[0] + 1)
    conv_width_2 = int(conv_width_1 / 2 - filter_2[1] + 1)

    if padding == "SAME":
        A2 = tf.reshape(A2, [int(img_height/2), int(img_width/2), batch_size*n_samples, n_filters_1, n_filters_2])
    if padding == "VALID":
        A2 = tf.reshape(A2, [conv_height_2, conv_width_2, batch_size*n_samples, n_filters_1, n_filters_2])

    A2 = tf.transpose(A2, [2, 0, 1, 3, 4])
    A2 = tf.reduce_sum(A2, axis=3)

    # Final activations for the first convolutional layer
    A2 = tf.nn.bias_add(A2, bias_C2)
    h2 = tf.nn.relu(A2)

    #---------#
    # POOLING #
    #---------#
    h2 = tf.nn.max_pool(h2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = padding)

    #------------#
    # FLATTENING #
    #------------#

    if padding == "SAME":
        flattened_shape = int((img_height * img_width) / 8 * n_filters_2)
    if padding == "VALID":
        flattened_shape = int((conv_width_2 * conv_height_2) / 4 * n_filters_2)

    h2 = tf.reshape(h2, shape = [n_samples, batch_size, flattened_shape, 1])        # The extra dimension (1) is needed for the FC layers
    h2 = tf.transpose(h2, [ 1, 0, 2, 3]) 

    ##########################
    # FULLY CONNECTED LAYERS #
    ##########################

    fullc_1 = dict_weights["d1"]
    fullc_out = dict_weights["out"]

    W_D1 = tf.reshape(pre_W_D1, shape = [ batch_size,  n_samples, fullc_1[0], fullc_1[1] ])
    W_OUT = tf.reshape(pre_W_OUT, shape = [ batch_size,  n_samples, fullc_out[0], fullc_out[1] ])

    # First FC layer
    A3 = tf.reduce_sum(h2 * W_D1, axis = 2) + bias_D1
    h3 = tf.nn.relu(A3)

    # DROPOUT
    if training == 1:
        h3 = tf.nn.dropout(h3, rate = dropout_rate)

    # Second FC layer
    A4 = tf.reduce_sum(tf.reshape(h3, shape = [ batch_size, n_samples, fullc_out[0], 1 ]) * W_OUT, axis = 2) + bias_OUT

    #######################
    # Objective functions #
    #######################

    probs = tf.nn.softmax(A4, axis = 2)
    pred_classes = tf.argmax(tf.reduce_mean(tf.nn.softmax(A4, axis = 2), axis = 1), axis = -1)

    tiled_y = tf.tile(tf.expand_dims(y_target, 0), [n_samples, 1, 1])
    tiled_y = tf.transpose(tiled_y, [1, 0, 2])
    labels = tf.argmax(tiled_y, 2)

    # Objective function

    res_train = tf.reduce_mean( -1.0 * tf.nn.sparse_softmax_cross_entropy_with_logits(labels = labels, logits = A4), axis = 1)


    # Test log loglikelihood

    log_prob_data = tf.reduce_sum(tf.reduce_logsumexp(-1.0 * tf.nn.sparse_softmax_cross_entropy_with_logits(\
        labels = tf.argmax(tiled_y, axis = 2), logits = A4), axis = 1) - tf.log(tf.cast(n_samples, tf.float32)))

    # Classification error

    class_error = 1.0 - tf.reduce_mean(tf.cast(tf.equal(tf.argmax(tf.reduce_mean(tf.nn.softmax(A4, axis = 2), axis = 1), \
        axis = 1), tf.argmax(y_target, axis = 1)), tf.float32))

    acc_op = 1.0 - class_error

    return res_train, log_prob_data, class_error, acc_op, pred_classes, tf.argmax(tiled_y, axis = 2)

###############################################################################
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
    x_test = mnist.test.images
    y_test = mnist.test.labels

    # Flatten the images into vectors
    dim_data = img_width * img_height
    X_train = x_train.reshape(x_train.shape[0], img_height, img_width, 1)
    X_test = x_test.reshape(x_test.shape[0], img_height, img_width, 1)

    total_training_data = x_train.shape[0]

    # convert class vectors to binary class matrices
    y_train = one_hot(y_train, num_classes)
    y_test = one_hot(y_test, num_classes)

    # Create the model
    # Placeholders for data and number of samples

    x = tf.placeholder(tf.float32, [ None, img_height, img_width, 1 ])
    y_ = tf.placeholder(tf.float32, [ None, num_classes ])
    n_samples = tf.placeholder(tf.int32, [ 1 ])[ 0 ]
    training = tf.placeholder(tf.int32, shape = [ 1 ])[ 0 ]

    # n_layers_nn = n_layers_gen = n_layers_disc = layers

    # Calculate the total number of parameters needed in the CNN
    total_weights_main = dict_weights.copy()
    total_weights_main.update((x, np.prod(y)) for x, y in total_weights_main.items())

    total_weights = sum(total_weights_main.values()) # + sum(dict_biases.values()) # The biases are deterministic

    # Create the networks that compose the system
    generator = create_generator(total_weights)
    main_CNN = create_main_CNN(dict_biases)

    # Output the weights sampled from the generator
    weights = compute_output_generator(generator, tf.shape(x)[ 0 ], n_samples, total_weights)

    KL = 0.5 * tf.reduce_sum(tf.exp(generator[ 'log_var_noise' ] - main_CNN['log_vars_prior']) + \
        (0.0 - generator[ 'mean_noise' ])**2 / tf.exp(main_CNN['log_vars_prior']) - 1.0 + main_CNN['log_vars_prior'] \
        - generator[ 'log_var_noise' ])

    # ==========================================================================
    # SEPARATE WEIGHTS IN DICTIONARY TO USE THEM IN THE CNN
    # ==========================================================================

    res_train, log_prob_data, classification_error, alt_accuracy, forecasts, real_labels = \
        compute_outputs_main_CNN(total_weights_main, dict_weights, dict_biases, x, y_, weights, main_CNN, n_samples, training)

    # Make the estimates of the ELBO for the primary classifier

    ELBO = (tf.reduce_sum(res_train) - tf.reduce_mean(KL) * tf.cast(tf.shape(x)[ 0 ], tf.float32) / \
        tf.cast(total_training_data, tf.float32)) * tf.cast(total_training_data, tf.float32) / tf.cast(tf.shape(x)[ 0 ], tf.float32)

    neg_ELBO = -ELBO
    main_loss = neg_ELBO
    mean_ELBO = ELBO

    # KL y res_train have shape batch_size x n_samples

    mean_KL = tf.reduce_mean(KL)

    # Create the variable lists to be updated

    vars_primal = get_variables_generator(generator) + get_variables_main_CNN(main_CNN)

    train_step_primal = tf.train.AdamOptimizer(primal_rate).minimize(main_loss, var_list = vars_primal)

    config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1, \
        allow_soft_placement=True, device_count = {'CPU': 1})

    with tf.Session(config = config) as sess:

        sess.run(tf.global_variables_initializer())

        total_ini = time.time()

        # Change the value of alpha to begin exploring using the second value given

        for epoch in range(n_epochs):

            L = 0.0
            class_error_estimate = 0.0
            kl = 0.0

            n_batches_train = int(np.ceil(total_training_data / n_batch))
            for i_batch in range(n_batches_train):

                ini = time.clock()
                ini_ref = time.time()
                ini_train = time.clock()

                last_point = np.minimum(n_batch * (i_batch + 1), total_training_data)

                batch = [ X_train[ i_batch * n_batch : last_point, :, :, :] , y_train[ i_batch * n_batch : last_point, : ] ]

                 # Dropout is turned off since it shows better results
                sess.run(train_step_primal, feed_dict={x: batch[ 0 ], y_: batch[ 1 ], n_samples: n_samples_train, \
                    training : 0})

                Ltemp, kltemp, class_error_train = sess.run([ mean_ELBO, mean_KL, \
                    classification_error], feed_dict={x: batch[ 0 ], y_: batch[ 1 ], n_samples: n_samples_train, training : 0})

                L += Ltemp
                kl += kltemp
                class_error_estimate += class_error_train

                fini = time.clock()
                fini_ref = time.time()
                fini_train = time.clock()

                sys.stdout.write('.')
                sys.stdout.flush()

                fini_train = time.clock()

                if ((i_batch + 1) % batches_to_report) == 0:
                    string = ('\n VI - batch %g datetime %s epoch %d ELBO %g  KL %g \
                        real_time %g cpu_time %g train_time %g') % \
                        (i_batch, str(datetime.now()), epoch, \
                        L / batches_to_report, kl / batches_to_report, (fini_ref - \
                        ini_ref), (fini - ini), (fini_train - ini_train))
                    print(string)
                    sys.stdout.flush()
                    string_acc = ('Error %g ') % (class_error_estimate / batches_to_report)
                    print(string_acc)

                    L = 0.0
                    kl = 0.0
                    class_error_estimate = 0.0


        # Test Evaluation

        sys.stdout.write('\n')
        ini_test = time.time()

        # We do the test evaluation RMSE

        cl_errors = 0.0
        LL  = 0.0
        acc = 0.0
        n_batches_to_process = int(np.ceil(X_test.shape[ 0 ] / n_batch_test))
        for i in range(n_batches_to_process):

            last_point = np.minimum(n_batch_test * (i + 1), X_test.shape[ 0 ])

            batch = [ X_test[ i * n_batch_test : last_point, : ] , y_test[ i * n_batch_test : last_point, ] ]

            LL_tmp, cl_error_tmp, acc_tmp = sess.run([ log_prob_data, classification_error, alt_accuracy ], feed_dict={x: batch[0], y_: batch[1], n_samples: n_samples_test, training : 0})
            cl_errors += cl_error_tmp
            LL += LL_tmp
            acc += acc_tmp

        error_class = cl_errors / float(X_test.shape[ 0 ])
        TestLL = (LL / float(X_test.shape[ 0 ]))
        accuracy_test = (acc / float(X_test.shape[ 0 ]))

        fini_test = time.time()
        fini = time.clock()
        fini_ref = time.time()
        total_fini = time.time()

        string = ('VI - batch %g datetime %s epoch %d ELBO %g  KL %g real_time %g cpu_time %g ' + \
            'test_time %g total_time %g LL %g Classification_error %g Alternative_accuracy %g') % \
            ( i_batch, str(datetime.now()), epoch, \
            L / n_batches_train, kl / n_batches_train, (fini_ref - \
            ini_ref), (fini - ini), (fini_test - ini_test), (total_fini - total_ini), \
            TestLL, error_class, accuracy_test)
        print(string)
        sys.stdout.flush()

        final_forecasts, final_labels = sess.run([ forecasts, real_labels ], feed_dict={x: X_test[0:10], y_: y_test[0:10], n_samples: n_samples_test, training : 0})


        np.savetxt('res_vi/results_error.txt', [ error_class ])
        np.savetxt('res_vi/results_ll.txt', [ TestLL ])
        np.savetxt('res_vi/results_accuracy.txt', [ accuracy_test ])
        np.savetxt('res_vi/forecasts.txt', final_forecasts)
        np.savetxt('res_vi/labels.txt', final_labels)




if __name__ == '__main__':

    # Create the folder to save all the results
    if not os.path.isdir("res_vi"):
        os.makedirs("res_vi")

    main( )


