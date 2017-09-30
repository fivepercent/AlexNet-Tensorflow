import sys
sys.path.append('../MNIST')
import os.path

import tensorflow as tf
import numpy as np
import loadData
import util
import AlexNet

def train_mnist(max_iter, learning_rate, f):
	resume=os.path.exists('./session/alexnet_mnist.meta')
	
	#Load data
	x_train_raw= loadData.load_train_images().reshape([-1,784])/255
	y_train_raw = loadData.load_train_labels()

	batch_size=10000
	data_length=x_train_raw.shape[0]
	num_batches=(int)(data_length/batch_size)

	x_train=tf.placeholder(tf.float32, [None, 784])
	y_train=tf.placeholder(tf.int32,[None])
	y_onehot=tf.one_hot(y_train, 10)
	x = tf.reshape(x_train,[-1,28,28,1])

	y=AlexNet.classifier(x)
	global_step = tf.Variable(0, trainable=False)
	lr = tf.placeholder(tf.float32)

	#cost function and accuracy
	loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_onehot, logits=y))
	train=tf.train.AdamOptimizer(learning_rate=lr).minimize(loss, global_step=global_step)
	accuracy= util.accuracy(y,y_onehot)

	config=tf.ConfigProto(allow_soft_placement= True,log_device_placement= True)
	saver = tf.train.Saver()
	sess=tf.Session(config=config)

	if resume:
		saver.restore(sess,'./session/alexnet_mnist')
	else:
		sess.run(tf.global_variables_initializer())

    #training
	step=0
	while step<max_iter:
		split=0
		for i in range(num_batches):
			x_b=x_train_raw[split:(split+batch_size)]
			y_b=y_train_raw[split:(split+batch_size)]
			split+=batch_size
			_, l, acc, gs = sess.run([train, loss, accuracy,global_step],feed_dict={lr:learning_rate, x_train:x_b, y_train:y_b})
			step=(gs-1)/6
			if(step%f==0):print ('Iter: %d, Loss: %f Acc: %f'%(step, l, acc))

    #Save model for continuous learning
	saver.save(sess, './session/alexnet_mnist')

	sess.close()