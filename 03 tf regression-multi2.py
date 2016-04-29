import tensorflow as tf

x_data = [[1,0,3,0,5],
          [0,2,0,4,0]]
y_data = [1,2,3,4,5]

W = tf.Variable(tf.random_uniform([1,2], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

### placeholder
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)


### hypothesis
hypothesis = tf.matmul(W, X) +b

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
        print step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W), sess.run(b)


print sess.run(hypothesis, feed_dict={X:[[5],[2]]})
print sess.run(hypothesis, feed_dict={X:[[2.5],[3]]})
