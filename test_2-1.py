import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1

def xavier_init(n_inputs, n_outputs, uniform=True):

    if uniform:

        init_range = tf.sqrt(6.0/ (n_inputs + n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)
    else:
        stddev = tf.sqrt(3.0/(n_inputs + n_outputs))
        return tf.truncated_normal_initializer(stddev = stddev)



# tf Graph input
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])
# ----------------이번에추가한부분----------------------
W1 = tf.get_variable("W1", shape = [784,256], initializer = xavier_init(784,256))
W2 = tf.get_variable("W2", shape = [256,256], initializer = xavier_init(256,256))
W3 = tf.get_variable("W3", shape = [256, 10], initializer = xavier_init(256,10))

B1 = tf.Variable(tf.zeros([256]))
B2 = tf.Variable(tf.zeros([256]))
B3 = tf.Variable(tf.zeros([10]))

L1 = tf.nn.relu(tf.add(tf.matmul(X,W1),B1))
L2 = tf.nn.relu(tf.add(tf.matmul(L1,W2),B2))
hypothesis = tf.add(tf.matmul(L2,W3),B3)


# -------------------------------------------------
# cross entropy
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = hypothesis, labels = Y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)

    #Training cycle starts
    for epoch in range(training_epochs):
        avg_cost = 0.

        total_batch = int(mnist.train.num_examples/batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)

            _, c= sess.run([optimizer, cost], feed_dict={X: batch_xs, Y:batch_ys})

            avg_cost += c/ total_batch

        if (epoch+1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    print("Optimization Finished!")

    correct_prediction = tf.equal(tf.argmax(hypothesis,1),tf.argmax(Y,1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels}))


