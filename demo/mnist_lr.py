import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


model_path = './models/test1.model.ckpt'
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# model
y = tf.matmul(x, W) + b

y_ = tf.placeholder(tf.float32, [None, 10])

# loss definition
loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)

# optimizer definition
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

with tf.Session() as sess:
    # variable initialize
    sess.run(tf.global_variables_initializer())

    # iteration
    for i in range(100):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

    saver = tf.train.Saver()
    saver.save(sess, model_path)
