import tensorflow as tf

def weights(shape,stddev,name):
    return tf.Variable(tf.truncated_normal(shape, stddev=stddev),name=name)
def bias(shape, name):
    return tf.Variable(tf.constant(0.0, shape=shape),name=name)
def convLayer(data_in, kernel, stride, padding):
    return tf.nn.conv2d(data_in, kernel,strides=[1, stride, stride, 1], padding=padding)
def max_pool(data_in, size, stride):
    return tf.nn.max_pool(data_in, ksize=[1, size, size, 1],strides=[1, stride, stride, 1], padding='SAME')


def accuracy(y,y_label):
	correct = tf.equal(tf.argmax(y, 1), tf.argmax(y_label, 1))
	accuracy= tf.reduce_mean(tf.cast(correct, tf.float32))
	return accuracy