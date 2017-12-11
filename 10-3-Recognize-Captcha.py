import os
import tensorflow as tf
from PIL import Image  # pip install Pillow
from nets import nets_factory
import numpy as np

CHAR_SET_LEN = 10
IMAGE_HEIGHT = 60
IMAGE_WIDTH = 160
BATCH_SIZE = 25
TFRECORD_FILE = "D:\Tensorflow\TF_basic\captcha\train.tfrecords"

x = tf.placeholder(tf.float32, [None, 224, 224])
y0 = tf.placeholder(tf.float32, [None])
y1 = tf.placeholder(tf.float32, [None])
y2 = tf.placeholder(tf.float32, [None])
y3 = tf.placeholder(tf.float32, [None])

lr = tf.Variable(0.003, dtype=tf.float32)


def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'image': tf.FixedLenFeature([], tf.string),
                                           'label0': tf.FixedLenFeature([], tf.int64),
                                           'label1': tf.FixedLenFeature([], tf.int64),
                                           'label2': tf.FixedLenFeature([], tf.int64),
                                           'label3': tf.FixedLenFeature([], tf.int64),
                                       })

    # get pic data
    image = tf.decode_raw(features['image'], tf.uint8)
    image = tf.reshape(image, [224, 224])
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)

    l0 = tf.cast(features['label0'], tf.int32)
    l1 = tf.cast(features['label1'], tf.int32)
    l2 = tf.cast(features['label2'], tf.int32)
    l3 = tf.cast(features['label3'], tf.int32)

    return image, l0, l1, l2, l3


image, label0, label1, label2, label3 = read_and_decode(TFRECORD_FILE)

image_batch, label_batch0, label_batch1, label_batch2, label_batch3 = tf.train.shuffle_batch(
    [image, label0, label1, label2, label3], batch_size=BATCH_SIZE,
    capacity=50000, min_after_dequeue=10000, num_threads=1
)

train_network_fn = nets_factory.get_network_fn(
    'alexnet_v2',
    num_classes=CHAR_SET_LEN,
    weight_decay=0.0005,
    is_training=True
)

with tf.Session()as sess:
    X = tf.reshape(x, [BATCH_SIZE, 224, 224, 1])
    logits0, logits1, logits2, logits3, end_points = train_network_fn(X)

    one_hot_labels0 = tf.one_hot(indices=tf.cast(y0, tf.int32), depth=CHAR_SET_LEN)
    one_hot_labels1 = tf.one_hot(indices=tf.cast(y1, tf.int32), depth=CHAR_SET_LEN)
    one_hot_labels2 = tf.one_hot(indices=tf.cast(y2, tf.int32), depth=CHAR_SET_LEN)
    one_hot_labels3 = tf.one_hot(indices=tf.cast(y3, tf.int32), depth=CHAR_SET_LEN)

    loss0 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits0, labels=one_hot_labels0))
    loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits1, labels=one_hot_labels1))
    loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits2, labels=one_hot_labels2))
    loss3 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits3, labels=one_hot_labels3))

    total_loss = (loss0 + loss1 + loss2 + loss3) / 4.0
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(total_loss)

    correct_prediction0 = tf.equal(tf.argmax(one_hot_labels0, 1), tf.argmax(logits0, 1))
    accuracy0 = tf.reduce_mean(tf.cast(correct_prediction0, tf.float32))

    correct_prediction1 = tf.equal(tf.argmax(one_hot_labels1, 1), tf.argmax(logits1, 1))
    accuracy1 = tf.reduce_mean(tf.cast(correct_prediction1, tf.float32))

    correct_prediction2 = tf.equal(tf.argmax(one_hot_labels2, 1), tf.argmax(logits2, 1))
    accuracy2 = tf.reduce_mean(tf.cast(correct_prediction2, tf.float32))

    correct_prediction3 = tf.equal(tf.argmax(one_hot_labels3, 1), tf.argmax(logits3, 1))
    accuracy3 = tf.reduce_mean(tf.cast(correct_prediction3, tf.float32))

    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(6001):
        b_image, b_label0, b_label1, b_label2, b_label3 = sess.run(
            [image_batch, label_batch0, label_batch1, label_batch2, label_batch3])
        sess.run(optimizer, feed_dict={x: b_image, y0: b_label0, y1: b_label1, y2: b_label2, y3: b_label3})

        if i % 20 == 0:
            if i % 2000 == 0:
                sess.run(tf.assign(lr, lr / 3))
            acc0, acc1, acc2, acc3, loss_ = sess.run([accuracy0, accuracy1, accuracy2, accuracy3, total_loss],
                                                     feed_dict={
                                                         x: b_image, y0: b_label0, y1: b_label1, y2: b_label2,
                                                         y3: b_label3
                                                     })

            learning_rate = sess.run(lr)
            print("Iter:%d Loss:%.3f Accuracy:%.2f, %.2f, %.2f, %.2f  Learning_rate:%.4f" % (
                i, loss_, acc0, acc1, acc2, acc3, learning_rate))

            if i == 6000:
                saver.save(sess, "./captcha/models/crack_captcha.model", global_step=1)
                break

    coord.request_stop()
    coord.join(threads)
