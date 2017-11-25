# homework

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# load dataset
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# define batch size
batch_size = 100;
n_batch = mnist.train.num_examples

x = tf.placeholder(tf.float32, [None, 784])  # 784=28x28 pixels in each picture
y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
lr = tf.Variable(0.001, dtype=tf.float32)

# create complex neuron network
W1 = tf.Variable(tf.truncated_normal([784, 500], stddev=0.1))
b1 = tf.Variable(tf.zeros([500]) + 0.1)
L1 = tf.nn.tanh(tf.matmul(x, W1) + b1)
L1_drop = tf.nn.dropout(L1, keep_prob)

W2 = tf.Variable(tf.truncated_normal([500, 300], stddev=0.1))
b2 = tf.Variable(tf.zeros([300]) + 0.1)
L2 = tf.nn.tanh(tf.matmul(L1_drop, W2) + b2)
L2_drop = tf.nn.dropout(L2, keep_prob)

W3 = tf.Variable(tf.truncated_normal([300, 10], stddev=0.1))
b3 = tf.Variable(tf.zeros([10]) + 0.1)
prediction = tf.nn.softmax(tf.matmul(L2_drop, W3) + b3)

# loss function (comparing square-mean with cross-entropy
# loss = tf.reduce_mean(tf.square(y - prediction))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))

# GDO
train_step = tf.train.AdamOptimizer(lr).minimize(loss)

init = tf.global_variables_initializer()

# save result
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
# get accuracy rate
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# run
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(51):
        sess.run(tf.assign(lr, 0.001 * (0.95 ** epoch)))  # 0.95**epoch means epoch's power of 0.95
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})

        learning_rate = sess.run(lr)
        test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
        print("iter " + str(epoch) + ", testing accuracy " + str(test_acc) + ",Learning rate:" + str(learning_rate))
