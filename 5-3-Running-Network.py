# Tensorboard


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from _datetime import datetime

# load dataset
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# define batch size
batch_size = 100;
n_batch = mnist.train.num_examples // batch_size


def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('stddev', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


# namespace
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x_input')  # 784=28x28 pixels in each picture
    y = tf.placeholder(tf.float32, [None, 10], name='y_input')

# create simple neuron network
with tf.name_scope('layer'):
    with tf.name_scope('weights'):
        W = tf.Variable(tf.zeros([784, 10]), name='W')
        variable_summaries(W)
    with tf.name_scope('bias'):
        b = tf.Variable(tf.zeros([10]), name='b')
        variable_summaries(b)
    with tf.name_scope('wx_plus_b'):
        wx_plus_b = tf.matmul(x, W) + b
    with tf.name_scope('softmax'):
        prediction = tf.nn.softmax(wx_plus_b)

# loss function (comparing square-mean with cross-entropy
# loss = tf.reduce_mean(tf.square(y - prediction))
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
    tf.summary.scalar('loss', loss)
# GDO
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

init = tf.global_variables_initializer()

# save result
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
    with tf.name_scope('accuracy'):
        # get accuracy rate
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

# timestamp
time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print(time)

# merge all summary
merged = tf.summary.merge_all()

# run
with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter('logs/', sess.graph)
    for epoch in range(3):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            summary, _ = sess.run([merged, train_step], feed_dict={x: batch_xs, y: batch_ys})

        writer.add_summary(summary, epoch)
        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("iter " + str(epoch) + ", testing accuracy " + str(acc))

        # timestamp
        time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(time)
