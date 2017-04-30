# imports
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# globals
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# C-style main function
def main():

    # placeholder for the images reshaped from [ 28 28 ] to [ 784 ]
    x = tf.placeholder(tf.float32, [None, 784])

    # placehodler for the correct classes
    y_ = tf.placeholder(tf.float32, [None, 10])


    # weights, initialized to all zeros
    W = tf.Variable(tf.zeros([784, 10]))

    # basises, initialized to all zeros
    b = tf.Variable(tf.zeros([10]))

    # predicted probabilities for each class
    y = tf.nn.softmax(tf.matmul(x, W) + b)


    # calculate the average cross_entropy between our predictions (y) and the correct classes (y_)
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

    # define the training step, which uses gradient descent to minimize the cross_entropy
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


    # create a session
    sess = tf.InteractiveSession()

    # initialize all the variables defined above
    tf.global_variables_initializer().run()

    # train 1000 batches of 100 images form the training data
    for _ in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict = {x: batch_xs, y_: batch_ys})


    # find which images the model correctly predicted 
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

    # reduce the vector of booleans to a percentage correct
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # print the results of the accuracy calculation
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

#main loop
if __name__ == '__main__':
    main()
