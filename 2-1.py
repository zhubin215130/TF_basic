#Graph
import  tensorflow as tf

#define graph
m1 = tf.constant([[3,3]])
m2 = tf.constant([[2],[3]])
product = tf.matmul(m1, m2)
#print(product)


#execute the graph
sess = tf.Session()
writer = tf.summary.FileWriter("D://tf-log/",sess.graph)
result=sess.run(product)
print(result)
sess.close()

with tf.Session() as sess:
    result=sess.run(product)
    print(result)
