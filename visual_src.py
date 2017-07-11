
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
print(mnist.train.labels[4])
X3 = mnist.train.images[4]
img3 = X3.reshape([28, 28])
plt.imshow(img3, cmap='gray')
plt.show()

X3.shape = [-1, 784]
y_batch = mnist.train.labels[0]

class_num = 10

y_batch.shape = [-1, class_num]

_X = tf.placeholder(tf.float32, [None, 784])


sess = tf.Session()
X3_outputs = np.array(sess.run(outputs, feed_dict={
            _X: X3, y: y_batch, keep_prob: 1.0, batch_size: 1}))
print (X3_outputs.shape)
X3_outputs.shape = [28, hidden_size]
print (X3_outputs.shape)
