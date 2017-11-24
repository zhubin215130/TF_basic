# MNIST-simple


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from _datetime import datetime

# load dataset
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)


# define batch size
batch_size = 100;
n_batch = mnist.train.num_examples

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

# timestamp
time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print(time)
# run
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(21):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})

        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("iter " + str(epoch) + ", testing accuracy " + str(acc))

        # timestamp
        time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(time)

# fine-tune params: batch_size, add hidden layer, initial value, loss function, learning rate, optimizer type, training epoch

# execution result of square-mean loss function
# iter 0, testing accuracy 0.9259
# iter 1, testing accuracy 0.9289
# iter 2, testing accuracy 0.9301
# iter 3, testing accuracy 0.9308
# iter 4, testing accuracy 0.9311
# iter 5, testing accuracy 0.9312
# iter 6, testing accuracy 0.9315
# iter 7, testing accuracy 0.9307
# iter 8, testing accuracy 0.9307
# iter 9, testing accuracy 0.9303
# iter 10, testing accuracy 0.9298
# iter 11, testing accuracy 0.9296
# iter 12, testing accuracy 0.9297
# iter 13, testing accuracy 0.9299
# iter 14, testing accuracy 0.9299
# iter 15, testing accuracy 0.93
# iter 16, testing accuracy 0.9306
# iter 17, testing accuracy 0.93
# iter 18, testing accuracy 0.9311
# iter 19, testing accuracy 0.9316
# iter 20, testing accuracy 0.9318


# execution result of cross entropy loss function
# iter 0, testing accuracy 0.9285
# iter 1, testing accuracy 0.9297
# iter 2, testing accuracy 0.93
# iter 3, testing accuracy 0.9301
# iter 4, testing accuracy 0.9299
# iter 5, testing accuracy 0.9299
# iter 6, testing accuracy 0.9297
# iter 7, testing accuracy 0.9302
# iter 8, testing accuracy 0.9305
# iter 9, testing accuracy 0.9298
# iter 10, testing accuracy 0.9298
# iter 11, testing accuracy 0.9294
# iter 12, testing accuracy 0.9299
# iter 13, testing accuracy 0.9295
# iter 14, testing accuracy 0.9294
# iter 15, testing accuracy 0.9295
# iter 16, testing accuracy 0.9293
# iter 17, testing accuracy 0.9292
# iter 18, testing accuracy 0.929
# iter 19, testing accuracy 0.9289
# iter 20, testing accuracy 0.9289
