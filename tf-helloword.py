import tensorflow as tf

h = tf.constant("Hello TF!")

with tf.Session() as sess:
    print(sess.run(h))
