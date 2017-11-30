# MNIST-simple


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from _datetime import datetime

# load dataset
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)


# define batch size
batch_size = 100;
n_batch = mnist.train.num_examples // batch_size

x = tf.placeholder(tf.float32, [None, 784])  # 784=28x28 pixels in each picture
y = tf.placeholder(tf.float32, [None, 10])

# create simple neuron network
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
prediction = tf.nn.softmax(tf.matmul(x, W) + b)

# loss function (comparing square-mean with cross-entropy
# loss = tf.reduce_mean(tf.square(y - prediction))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))

# GDO
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

init = tf.global_variables_initializer()

# save result
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
# get accuracy rate
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver=tf.train.Saver()

# timestamp
time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print(time)
# run
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels}))
    saver.restore(sess,'net/my_mnist_net.ckpt')
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
