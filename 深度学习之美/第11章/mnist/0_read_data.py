'''
根据《深度学习之美》张玉宏的11章，将TensorFlow的内容，自己手动编写的代码
2019年11月
'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import ssl
import matplotlib.pyplot as plt

print("tensorflow version:", tf.__version__) # 1.2.0
# 全局取消证书验证，以解决：[SSL: CERTIFICATE_VERIFY_FAILED]
ssl._create_default_https_context = ssl._create_unverified_context

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

print(mnist.train.images.shape, mnist.train.labels.shape)
print(mnist.test.images.shape, mnist.test.labels.shape)
print(mnist.validation.images.shape, mnist.validation.labels.shape)

n_samples = 5
plt.figure(figsize=(n_samples * 2, 3))
for index in range(n_samples):
    plt.subplot(1, n_samples, index + 1)
    sample_image = mnist.train.images[index].reshape(28, 28)
    plt.imshow(sample_image, cmap="binary")
    plt.axis("off")

with tf.Session() as sess:
    print(sess.run(tf.argmax(mnist.train.labels[:n_samples], 1)))
plt.show()
