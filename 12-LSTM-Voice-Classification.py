import os
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
import librosa  # pip install librosa
from tqdm import tqdm  # pip install tqdm  for progress bar
import random

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .2, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("parent_dir", "audio/", "Data source for the positive data.")
tf.flags.DEFINE_string("tr_sub_dirs", ['fold1/', 'fold2/', 'fold3/'], "Data source for the negative data.")

# Model Hyperparameters
tf.flags.DEFINE_integer("n_inputs", 40, "Number of MFCCs, default is 40")
tf.flags.DEFINE_string("n_hidden", 300, "Number of cells, default is 300")
tf.flags.DEFINE_integer("n_classes", 10, "Number of classes, default is 10")
tf.flags.DEFINE_integer("lr", 0.005, "Dropout keep probability, default is 0.5")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 50, "Batch Size (default: 50)")
tf.flags.DEFINE_integer("num_epochs", 100, "Number of training epochs (default: 100)")
tf.flags.DEFINE_integer("evaluate_every", 50, "Evaluate model on dev set after this many steps (default: 50)")
tf.flags.DEFINE_integer("checkpoint_every", 500, "Save model after this many steps (default: 500)")
tf.flags.DEFINE_integer("num_checkpoints", 2, "Number of checkpoints to store (default: 2)")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


def get_wav_files(parent_dir, sub_dirs):
    wav_files = []
    for l, sub_dir in enumerate(sub_dirs):
        wav_path = os.path.join(parent_dir, sub_dir)
        for (dirpath, dirnames, filenames) in os.walk(wav_path):
            for filename in filenames:
                if filename.endswith('.wav') or filename.endswith('.WAV'):
                    filename_path = os.sep.join([dirpath, filename])
                    wav_files.append(filename_path)
    return wav_files


def extract_features(wav_files):
    inputs = []
    labels = []

    for wav_file in tqdm(wav_files):
        audio, fs = librosa.load(wav_file)
        mfccs = np.transpose(librosa.feature.mfcc(y=audio, sr=fs, n_mfcc=FLAGS.n_inputs), [1, 0])
        inputs.append(mfccs.tolist())

    for wav_file in wav_files:
        label = wav_file.split('/')[-1].split('-')[1]
        labels.append(label)
    return inputs, np.array(labels, dtype=np.int)


wav_files = get_wav_files(FLAGS.parent_dir, FLAGS.tr_sub_dirs)
tr_features, tr_labels = extract_features(wav_files)
np.save('tr_features.npy', tr_features)
np.save('tr_labels.npy', tr_labels)

# tr_features = np.load('tr_features.npy')
# tr_labels = np.load('tr_labels.npy')

wav_max_len = max([len(feature) for feature in tr_features])
print('max_len:', wav_max_len)

tr_data = []
for mfccs in tr_features:
    while len(mfccs) < wav_max_len:
        mfccs.append([0] * FLAGS.n_inputs)
        tr_data.append(mfccs)

tr_data = np.array(tr_data)

np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(tr_data)))
x_shuffled = tr_data[shuffle_indices]
y_shuffled = tr_labels[shuffle_indices]

dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y_shuffled)))
train_x, test_x = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
train_y, test_y = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

x = tf.placeholder('float', [None, wav_max_len, FLAGS.n_inputs])
y = tf.placeholder('float', [None])
dropout = tf.placeholder(tf.float32)
lr = tf.Variable(FLAGS.lr, dtype=tf.float32, trainable=False)

weights = tf.Variable(tf.truncated_normal([FLAGS.n_hidden, FLAGS.n_classes], stddev=0.1))
biases = tf.Variable(tf.constant(0.1, shape=[FLAGS.n_classes]))

num_layers = 3


def grucell():
    cell = tf.contrib.rnn.GRUCell(FLAGS.n_hidden)
    cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=dropout)
    return cell


cell = tf.contrib.rnn.MultiRNNCell([grucell() for _ in range(num_layers)])

outputs, final_state = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)

prediction = tf.nn.softmax(tf.matmul(final_state[0], weights) + biases)
one_hot_labels = tf.one_hot(indices=tf.cast(y, tf.int32), depth=FLAGS.n_classes)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=one_hot_labels))
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cross_entropy)

correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(one_hot_labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data) - 1 / batch_size) + 1
    print('num_batches_per_epoch:', num_batches_per_epoch)
    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffle_data = data[shuffle_indices]
        else:
            shuffle_data = data

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffle_data[start_index:end_index]


init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)

    batches = batch_iter(list(zip(train_x, train_y)), FLAGS.batch_size, FLAGS.num_epochs)

    for i, batch in enumerate(batches):
        i = i + 1
        x_batch, y_batch = zip(*batch)
        sess.run([optimizer], feed_dict={x: x_batch, y: y_batch, dropout: FLAGS.dropout_keep_prob})

        if i % FLAGS.evaluate_every == 0:
            sess.run(tf.assign(lr, FLAGS.lr * (0.99 ** (i // FLAGS.evaluate_every))))
            learning_rate = sess.run(lr)
            tr_acc, _loss = sess.run([accuracy, cross_entropy], feed_dict={x: train_x, y: train_y, dropout: 1.0})
            ts_acc = sess.run(accuracy, feed_dict={x: test_x, y: test_y, dropout: 1.0})
            print("Iter{} , loss {:.5f},tr_acc {:.5f}, ts_acc {:.5f}, lr {:.5f}".format(i, _loss, tr_acc, ts_acc,
                                                                                        learning_rate))

        if i % FLAGS.checkpoint_every == 0:
            path = saver.save(sess, "sounds_models/model", global_step=i)
            print('Saved model checkpoint to {}\n'.format(path))
