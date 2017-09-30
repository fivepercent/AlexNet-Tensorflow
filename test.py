import sys
sys.path.append('../MNIST')

import tensorflow as tf
import numpy as np
import loadData
import util
import AlexNet

def test_accuracy():
	x_test=tf.cast(loadData.load_test_images().reshape(-1,784),dtype='float32')
	y_test_raw = loadData.load_test_labels()
	y_test=tf.one_hot(y_test_raw, 10)
	x = tf.reshape(x_test,[-1,28,28,1])

	y=AlexNet.classifier(x)
	saver = tf.train.Saver()
	sess=tf.Session()
	saver.restore(sess,'./session/alexnet')
	accuracy=util.accuracy(y, y_test)
	acc=sess.run(accuracy)
	print('Accuracy for test data is: %f'%(acc))
	sess.close()