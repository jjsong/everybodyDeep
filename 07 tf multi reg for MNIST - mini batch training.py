import tensorflow as tf
import numpy as np
import input_data
from sklearn.metrics import *


### Load data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels


### Parameters
learning_rate = 0.1
training_epochs = 20
batch_size = 128
display_step = 1


### tf graph input : placeholder
X = tf.placeholder("float", [None, 784])   # 28x28= 784
Y = tf.placeholder("float", [None, 10])    # 0~9 = 10 classes

### Set model weights
W = tf.Variable(tf.zeros([784, 10]))  #weight's matrix 784 x and 10 y
b = tf.Variable(tf.zeros([10]))

### Construct model
activation = tf.nn.softmax(tf.matmul(X, W) + b)

### Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(activation), reduction_indices=1))
#learning_rate = tf.Variable(0.001)  #learning rate, alpha

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

### before starting, initialize the variables
init = tf.initialize_all_variables()

### lunch the graph
with tf.Session() as sess:
    sess.run(init)

    for epoch in range(training_epochs):
    	avg_cost = 0.
    	total_batch = int(mnist.train.num_examples/batch_size)

        #Loop over all batches
    	for i in range(total_batch):
            #mini-batch training  
    		batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            
            #Fit training using batch data
    		sess.run(optimizer, feed_dict={X:batch_xs, Y: batch_ys})
            #Comput average loss
    		avg_cost += sess.run(cost, feed_dict={X: batch_xs, Y: batch_ys}) / total_batch
        #Display logs per epoch step
    	if epoch % display_step ==0:
    		print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost)
    	print "Optimization Finished"

        #Get one and predict 
        # r = randint(0, mnist.test.num_examples -1) 
        # print "Label: ", sess.run(tf.argmax(minist.test.labels[r:r+1], 1))        
        # print "Prediction: ", sess.run(tf.argmax(activation, 1), {X:minist.test.images[r:r+1]})

        # #show the image
        # plt.imshow(mnist.test.images[r:r+1].reshape(28, 28), cmap='Greys', interpolation='nearest')
        # plt.show()


    	#Test Model
    	correct_prediction = tf.equal(tf.argmax(activation, 1), tf.argmax(Y, 1))
    	#Calculate accuracy
    	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    	print "Accuracy:", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels})
        #print confusion_matrix(tf.argmax(activation, 1), tf.argmax(Y, 1))
        #print "f1_score", f1_score(tf.argmax(activation, 1), tf.argmax(mnist.test.labels, 1))
        print ""

    	### confusion matrix :
        ### http://stackoverflow.com/questions/35365007/tensorflow-precision-recall-f1-score-and-confusion-matrix
        ### http://stackoverflow.com/questions/35756710/how-do-i-create-confusion-matrix-of-predicted-and-ground-truth-labels-with-tenso
    	#print sk.metrics.confusion_matrix(y_true, y_pred) fpr, tpr, tresholds = sk.metrics.roc_curve(y_true, y_pred)
