## imports
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

## globals
mnist = input_data.read_data_sets("MnIST_data/", one_hot=True)



## helper functions

# create a weight variable with shape: `shape`, with a small amount of noise for symmetry breaking
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# create a bias variable with shape: `shape`, biased positive to avoid dead ReLU neurons
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# convolve input x with weights W
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# return the maximum value in every 2x2 patch of the input, input [ n h w c ] -> output [ n h/2 w/2 c ]
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='SAME')

## C-style main function
def main():

    ### Creating the neural net
    ## input placeholers
    # placeholder for the images of shape [n 784]
    x = tf.placeholder(tf.float32, [None, 784])

    # placehodler for the correct classes
    y_ = tf.placeholder(tf.float32, [None, 10])

    # reshape the input into images, in form [n h w c]
    x_image = tf.reshape(x, [-1, 28, 28, 1])



    ## convolutional layer 1, [n 28 28 1] -> [n 14 14 32]
    # create weights for the first convolution (5x5 patch with 1 input and 32 outputs)
    W_conv1 = weight_variable([5, 5, 1, 32])

    # create biases associated with each of the 32 outputs of the convolution
    b_conv1 = bias_variable([32])

    # convolve input images with weights and add bias, x_image [n 28 28 1] -> h_conv1 [n 28 28 32]
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    # max pool the output convolutions, h_conv1 [n 28 28 32] -> h_pool1 [n 14 14 32]
    h_pool1 = max_pool_2x2(h_conv1)



    ## convolutional layer 2, [n 14 14 32] -> [n 7 7 64]
    # create weights for the second convolution (5x5 patch with 32 inputs and 64 outputs)
    W_conv2 = weight_variable([5, 5, 32, 64])

    # create biases associated with each of the 64 outputs of the convolution
    b_conv2 = bias_variable([64])

    # convolve input with weights and add bias, h_pool1 [n 14 14 32] -> h_conv2 [n 14 14 64]
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    # max pool the output convolutions, h_conv2 [n 14 14 64] -> h_pool2 [n 7 7 64]
    h_pool2 = max_pool_2x2(h_conv2)



    ## densely connected layer, [n 7 7 64] ->
    # create weights for the fully connected layer (7 * 7 * 64 inputs and 1024 outputs)
    W_fc1 = weight_variable([7 * 7 * 64, 1024])

    # create biases associated with each of the 1024 outputs of the fully connected layer
    b_fc1 = bias_variable([1024])

    # reshape the result of h_pool2 to be used in the fully connected layer [n 7 7 64] -> [n 7*7*64]
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

    # multiply reshaped result of the last layer by the weight matrix, add a bias, and apply a ReLU
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)



    ## dropout
    # placeholder probability to keep a node in the network, this so we can use dropout on during training, and turn it off during testing.
    keep_prob = tf.placeholder(tf.float32)

    # dropout the nodes related to the output of the densely connected layer with the probability: `keep_prob`
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)



    ## readout layer (densely connected layer 2)
    ## this layer brings the 1024 features of fc1 down to 10
    # create weights for the fully connected layer (1024 inputs and 10 outputs)
    W_fc2 = weight_variable([1024, 10])
    
    # create biases associated with each of the 10 outputs of the fully connected layer
    b_fc2 = bias_variable([10])

    # unnormalized predicted probabilities for each class from the convolutional nueral net
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2



    ### training/testing
    with tf.Session() as sess:
        ## creating the training/testing objects
        # calculate the average cross_entropy between our predictions (y_conv) and the correct classes (y_)
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

        # define the training step, which uses an ADAM optimizer to minimize the cross_entropy
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

        # find which images the model correctly predicted 
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

        # reduce the vector of booleans to a percentage correct
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        tf.summary.scalar('accuracy', accuracy)


        ## tensor board
        # merge all summarys
        merged = tf.summary.merge_all()

        # create a summary file writer
        writer = tf.summary.FileWriter('/tmp/tensorflow/mnist/logs/', sess.graph)


        ## train then test
        sess.run(tf.global_variables_initializer())
        
        for i in range(20000):
            batch = mnist.train.next_batch(50)
            
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})

                summary, train_accuracy = sess.run([merged, accuracy], feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
                writer.add_summary(summary, i)

                print('step %d, training accuracy %g' % (i, train_accuracy))

            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

        print('test accuracy %g' % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

        writer.close()



## 
if __name__ == '__main__':
    main()
