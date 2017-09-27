
# coding: utf-8

# In[6]:

import tensorflow as tf
import numpy as np
import tflearn.datasets.oxflower17 as oxflower17
import time


# In[7]:

data, label = oxflower17.load_data(one_hot=True)


# In[8]:

num_samples=data.shape[0]
split = int(num_samples * 0)
x_train=data[:split]
y_train=label[:split]
x_test=data[split:]
y_test=label[split:]


# In[9]:

def weights(shape,stddev,name):
    return tf.Variable(tf.truncated_normal(shape, stddev=stddev),name=name)
def bias(shape, name):
    return tf.Variable(tf.constant(0.0, shape=shape),name=name)
def convLayer(data_in, kernel, stride, padding):
    return tf.nn.conv2d(data_in, kernel,strides=[1, stride, stride, 1], padding=padding)
def max_pool(data_in, size, stride):
    return tf.nn.max_pool(data_in, ksize=[1, size, size, 1],strides=[1, stride, stride, 1], padding='SAME')


# In[10]:

#parameters for convolution kernels
IMG_NUM,IMG_HEIGHT,IMG_WIDTH,IMG_DEPTH=x_train.shape
C1_KERNEL_SIZE, C2_KERNEL_SIZE, C3_KERNEL_SIZE, C4_KERNEL_SIZE, C5_KERNEL_SIZE=11,5,3,3,3
C1_OUT_CHANNELS,C2_OUT_CHANNELS,C3_OUT_CHANNELS,C4_OUT_CHANNELS,C5_OUT_CHANNELS=96,256,384,384,256
#C1_OUT_CHANNELS,C2_OUT_CHANNELS,C3_OUT_CHANNELS,C4_OUT_CHANNELS,C5_OUT_CHANNELS=48,128,197,197,128
C1_STRIDES,C2_STRIDES,C3_STRIDES,C4_STRIDES,C5_STRIDES=4,1,1,1,1
P1_SIZE,P2_SIZE,P5_SIZE=3,3,3
P1_STRIDE,P2_STRIDE,P5_STRIDE=2,2,2
F6_SIZE,F7_SIZE=4096,4096
F8_SIZE=(int)(y_test.shape[1])


# In[11]:

#Load AlexNet Model
saver=tf.train.import_meta_graph('AlexNet-largeTrain-350.meta')
graph = tf.get_default_graph()
sess=tf.Session()
sess.run(tf.global_variables_initializer())
saver.restore(sess,'AlexNet-largeTrain-350')
C1_kernel=graph.get_tensor_by_name('C1_kernel:0')
C2_kernel=graph.get_tensor_by_name('C2_kernel:0')
C3_kernel=graph.get_tensor_by_name('C3_kernel:0')
C4_kernel=graph.get_tensor_by_name('C4_kernel:0')
C5_kernel=graph.get_tensor_by_name('C5_kernel:0')

C1_bias=graph.get_tensor_by_name('C1_bias:0')
C2_bias=graph.get_tensor_by_name('C2_bias:0')
C3_bias=graph.get_tensor_by_name('C3_bias:0')
C4_bias=graph.get_tensor_by_name('C4_bias:0')
C5_bias=graph.get_tensor_by_name('C5_bias:0')

F6_weights=graph.get_tensor_by_name('F6_weights:0')
F6_bias=graph.get_tensor_by_name('F6_bias:0')
F7_weights=graph.get_tensor_by_name('F7_weights:0')
F7_bias=graph.get_tensor_by_name('F7_bias:0')
F8_weights=graph.get_tensor_by_name('F8_weights:0')
F8_bias=graph.get_tensor_by_name('F8_bias:0')


# In[12]:

#AlexNet network

#Conv layer 1
C1=convLayer(x_test, C1_kernel, C1_STRIDES, 'SAME')
ReLU1=tf.nn.relu(C1+C1_bias)
P1=max_pool(ReLU1,P1_SIZE, P1_STRIDE)
NORM1=tf.nn.local_response_normalization(P1)

#Conv layer 2
C2=convLayer(NORM1, C2_kernel, C2_STRIDES, 'SAME')
ReLU2=tf.nn.relu(C2+C2_bias)
P2=max_pool(ReLU2,P2_SIZE, P2_STRIDE)
NORM2=tf.nn.local_response_normalization(P2)

#Conv layer 3
C3=convLayer(NORM2, C3_kernel, C3_STRIDES, 'SAME')
ReLU3=tf.nn.relu(C3+C3_bias)

#Conv layer 4
C4=convLayer(ReLU3, C4_kernel, C4_STRIDES, 'SAME')
ReLU4=tf.nn.relu(C4+C4_bias)

#Conv layer 5
C5=convLayer(ReLU4, C5_kernel, C5_STRIDES, 'SAME')
ReLU5=tf.nn.relu(C5+C5_bias)
P5_pre=max_pool(ReLU5,P5_SIZE, P5_STRIDE)

num_P5_out=(int)(P5_pre.shape[1]*P5_pre.shape[2]*P5_pre.shape[3])
P5=tf.reshape(P5_pre,[-1,num_P5_out])

#Fully connected layer 6

F6=tf.matmul(P5, F6_weights)+F6_bias
ReLU6=tf.nn.relu(F6)
DROP6=tf.nn.dropout(ReLU6, 0.5)

#Fully connected layer 7

F7=tf.matmul(DROP6, F7_weights)+F7_bias
ReLU7=tf.nn.relu(F7)
DROP7=tf.nn.dropout(ReLU7, 0.5)

#Fully connected layer 8

F8=tf.matmul(DROP7, F8_weights)+F8_bias
y=tf.nn.softmax(F8)


# In[13]:

print(C1_kernel)


# In[14]:

correct = tf.equal(tf.argmax(y, 1), tf.argmax(y_test, 1))
accuracy= tf.reduce_mean(tf.cast(correct, tf.float32))


# In[15]:

acc=sess.run(accuracy)
print(acc)


# In[ ]:

sess.close()

