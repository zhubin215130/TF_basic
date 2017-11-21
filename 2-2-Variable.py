#Variable

import tensorflow as tf
x=tf.Variable([1,2])
a=tf.constant([3,3])

sub =tf.subtract(x,a)
add=tf.add(x,sub)


#create a var and update its value
state=tf.Variable(0,name='counter')
new_value=tf.add(state,1)
update=tf.assign(state,new_value)


init = tf.global_variables_initializer()

with tf.Session() as sess:
    writer = tf.summary.FileWriter("D://tf-log/",sess.graph)
    sess.run(init)
    print(sess.run(sub))
    print(sess.run(add))
    
    print(sess.run(state))
    for _ in range(5):
        sess.run(update)
        print(sess.run(state))
