import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str,
                    default='./MNIST_data',
                    help='Directory for storing input data')
FLAGS, unparsed = parser.parse_known_args()

model_path = '../demo/models/test1.model.ckpt'

mnist = input_data.read_data_sets(FLAGS.data_dir)

x = tf.placeholder(tf.float32, [None, 784])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.matmul(x, W) + b

with tf.Session() as sess:
    saver = tf.train.Saver()
    saver.restore(sess, model_path)
    batch = mnist.test.next_batch(10)
    result = sess.run(y, feed_dict={x: batch[0]})
    print("real class: %s, predict class: %s" % (batch[1], tf.argmax(result, 1).eval()))
