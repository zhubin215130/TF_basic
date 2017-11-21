#Tensorflow tutorial: basic linear regression
import tensorflow as tf
import numpy as np

#use numpy generate 100 random points as training set
x_data=np.random.rand(100)
y_data=x_data*0.1+0.2

#construct linear model
b=tf.Variable(0.1)
k=tf.Variable(0.5)
y=k*x_data+b

#loss function
loss=tf.reduce_mean(tf.square(y_data-y))
#define GDO
optimizer=tf.train.GradientDescentOptimizer(0.2)
#Minimize loss
train=optimizer.minimize(loss)

init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for step in range(201):
        sess.run(train)
        if step%20 == 0:
            print(step, sess.run([k,b]))

# note: k and b are not stable in each run
