import tensorflow as tf

import numpy as np
xy = np.loadtxt('data/train-soft.txt', unpack=True, dtype='float32')

x_data = xy[0:3]
y_data = xy[3:]

print 'x', x_data
print 'y', y_data

#to make 'matmul (X, W)'
print ""
x_data = np.transpose(xy[0:3])
y_data = np.transpose(xy[3:])

print 'x', x_data
print 'y', y_data


### tf graph input : placeholder
X = tf.placeholder("float", [None, 3]) #x1, x2 and 1 (for bias b)
Y = tf.placeholder("float", [None, 3]) #A, B, C  => 3 classes

### Set model weights
W = tf.Variable(tf.zeros([3, 3]))  #weight's matrix 3 x and 3 y 


### hyothesis -> matmul -> softmax -> cross-entropy
hypothesis = tf.matmul(X, W)    #tf.matmul(X, W) 
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(hypothesis, Y))


######### same as the above
### Construct model :  hypothesis
#hypothesis = tf.nn.softmax(tf.matmul(X, W)) #softmax  # X * W 

### cost functhion : cross-entropy
#cost = -tf.reduce_mean(Y*tf.log(hypothesis) + (1-Y) * tf.log(1-hypothesis))
#cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), reduction_indices=1))
##########



### minimize : Gradient descent
learning_rate = tf.Variable(0.001)  #learning rate, alpha

optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(cost)


### before starting, initialize the variables
init = tf.initialize_all_variables()

### lunch the graph
with tf.Session() as sess:
    sess.run(init)

    ### fit the line
    for step in xrange(801):
        sess.run(train, feed_dict={X:x_data, Y:y_data})
        if step % 200 == 0:
            print step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W)

    a = sess.run(hypothesis, feed_dict={X: [[1, 11, 7]]})
    print a, sess.run(tf.arg_max(a,1))
    print ""

    b = sess.run(hypothesis, feed_dict={X: [[1,3, 4]]})
    print b, sess.run(tf.arg_max(b, 1))
    print ""

    c = sess.run(hypothesis, feed_dict={X: [[1, 1, 0]]})
    print c, sess.run(tf.arg_max(c, 1))
    print ""

    all = sess.run(hypothesis, feed_dict={X: [[1,11, 7], [1, 3, 4], [1, 1, 0]]})
    print all, sess.run(tf.arg_max(all, 1))



