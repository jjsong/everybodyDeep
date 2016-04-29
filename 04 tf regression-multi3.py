import tensorflow as tf


import numpy as np
xy = np.loadtxt('data/train-reg.txt', unpack=True, dtype='float32')

'''
??????
import pandas as pd
xy = pd.read_csv("data/train.csv", dtype='float32')

print ""
print xy[0:1]
print ""
print xy[0:2]
print ""
print xy[0:3]
print ""
print xy[0:-1]

print ""
print xy[-1]
???????
'''


x_data = xy[0:-1]
y_data = xy[-1]

print 'x', x_data
print 'y', y_data




W = tf.Variable(tf.random_uniform([1,3], -1.0, 1.0))
#b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

### placeholder
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)


### hypothesis
hypothesis = tf.matmul(W, X)

### simplified cost functhion
cost = tf.reduce_mean(tf.square(hypothesis - Y))

### minimize
a = tf.Variable(0.1)  #learning rate, alpha
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

### before starting, initialize the variables
init = tf.initialize_all_variables()

### lunch the graph
sess = tf.Session()
sess.run(init)


### fit the line
for step in xrange(2001):
    sess.run(train, feed_dict={X:x_data, Y:y_data})
    if step % 20 == 0:
        print step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W)  #, sess.run(b)


print sess.run(hypothesis, feed_dict={X:[[1],[5],[2]]})
#print sess.run(hypothesis, feed_dict={X:[[2.5],[3]]})
