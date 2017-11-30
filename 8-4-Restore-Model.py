# MNIST-simple


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from _datetime import datetime

# load dataset
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

y = tf.placeholder(tf.float32, [None, 10])

#load model
with tf.gfile.FastGFile('./models/my-mnist-model.pb','rb') as f:
    graph_def=tf.GraphDef();
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def,name='')



# timestamp
time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print(time)
# run
with tf.Session() as sess:
    output = sess.graph.get_tensor_by_name('output:0')

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(output, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy,feed_dict={'x-input:0':mnist.test.images,y:mnist.test.labels}))

# timestamp
time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print(time)