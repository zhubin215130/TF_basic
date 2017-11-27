import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from _datetime import datetime


mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

n_inputs = 28
max_time = 28
lstm_size = 100
n_classes = 10

batch_size = 50
n_batch = mnist.train.num_examples // batch_size

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

weights = tf.Variable(tf.truncated_normal([lstm_size, n_classes], stddev=0.1))
biases = tf.Variable(tf.constant(0.1, shape=[n_classes]))


def RNN(X, weights, biases):
    #inputs:[batch_size, max_time, n_inputs]
    inputs = tf.reshape(X, [-1, max_time, n_inputs])
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_size)
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, inputs, dtype=tf.float32)
    results = tf.nn.softmax(tf.matmul(final_state[1], weights) + biases)
    return results


prediction = RNN(x, weights, biases)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# timestamp
time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print(time)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(6):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})

        # timestamp
        time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(time)
        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("iter " + str(epoch) + ", testing accuracy " + str(acc))
