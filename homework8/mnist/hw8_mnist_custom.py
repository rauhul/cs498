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

## convolutional layer 1, [n h w inputs] -> [n h w outputs]
def conv_layer(x, kernal_size, inputs, outputs):
    # create kernal for the first convolution 
    W = weight_variable([kernal_size, kernal_size, inputs, outputs])

    # create biases associated with each of the outputs of the convolution
    b = bias_variable([outputs])

    # convolve input images with weights and add bias
    return tf.nn.relu(conv2d(x, W) + b)

def fc_layer(x, height, width, inputs, outputs):
    W = weight_variable([height * width * inputs, outputs])

    # create biases associated with each of the outputs of the fully connected layer
    b = bias_variable([outputs])

    # reshape the result to be used in the fully connected layer
    x_flat = tf.reshape(x, [-1, height * width * inputs])

    # multiply reshaped result of the last layer by the weight matrix, add a bias, and apply a ReLU
    return tf.nn.relu(tf.matmul(x_flat, W) + b)


## C-style main function
def main():

    ### Creating the neural net
    ## input placeholers
    # placeholder for the images of shape [n 784]
    x = tf.placeholder(tf.float32, [None, 784])

    # placehodler for the correct classes
    y_ = tf.placeholder(tf.float32, [None, 10])

    # placeholder probability to keep a node in the network, this so we can use dropout on during training, and turn it off during testing.
    keep_prob = tf.placeholder(tf.float32)

    # reshape the input into images, in form [n h w c]
    x_image = tf.reshape(x, [-1, 28, 28, 1])



    x_conv1 = conv_layer(x_image, 5, 1,  8)

    x_conv2 = conv_layer(x_conv1, 5, 8,  8)

    x_conv3 = conv_layer(x_conv2, 5, 8, 16)

    x_fc1 = fc_layer(x_conv3, 28, 28, 16, 512)
    
    x_fc1_drop = tf.nn.dropout(x_fc1, keep_prob)



    ## readout layer (densely connected layer 2)
    ## this layer brings the 1024 features of fc1 down to 10
    # create weights for the fully connected layer (1024 inputs and 10 outputs)
    W = weight_variable([512, 10])
    
    # create biases associated with each of the 10 outputs of the fully connected layer
    b = bias_variable([10])

    # unnormalized predicted probabilities for each class from the convolutional nueral net
    y = tf.matmul(x_fc1_drop, W) + b



    ### training/testing
    with tf.Session() as sess:
        ## creating the training/testing objects
        # calculate the average cross_entropy between our predictions (y) and the correct classes (y_)
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

        # define the training step, which uses an ADAM optimizer to minimize the cross_entropy
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

        # find which images the model correctly predicted 
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

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
        
        for i in range(2000):
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
