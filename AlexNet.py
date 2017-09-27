
# coding: utf-8

# In[1]:

import tensorflow as tf
import numpy as np
import tflearn.datasets.oxflower17 as oxflower17
import time


# In[2]:

data, label = oxflower17.load_data(one_hot=True)


# In[3]:

print(data.shape)


# In[4]:

num_samples=data.shape[0]
split = int(num_samples * 9/10)
x_train=data[:split]
y_train=label[:split]
x_test=data[split:]
y_test=label[split:]


# In[5]:

def weights(shape,stddev,name):
    return tf.Variable(tf.truncated_normal(shape, stddev=stddev),name=name)
def bias(shape, name):
    return tf.Variable(tf.constant(0.0, shape=shape),name=name)
def convLayer(data_in, kernel, stride, padding):
    return tf.nn.conv2d(data_in, kernel,strides=[1, stride, stride, 1], padding=padding)
def max_pool(data_in, size, stride):
    return tf.nn.max_pool(data_in, ksize=[1, size, size, 1],strides=[1, stride, stride, 1], padding='SAME')


# In[6]:

#parameters for convolution kernels
IMG_NUM,IMG_HEIGHT,IMG_WIDTH,IMG_DEPTH=x_train.shape
C1_KERNEL_SIZE, C2_KERNEL_SIZE, C3_KERNEL_SIZE, C4_KERNEL_SIZE, C5_KERNEL_SIZE=11,5,3,3,3
C1_OUT_CHANNELS,C2_OUT_CHANNELS,C3_OUT_CHANNELS,C4_OUT_CHANNELS,C5_OUT_CHANNELS=96,256,384,384,256
#C1_OUT_CHANNELS,C2_OUT_CHANNELS,C3_OUT_CHANNELS,C4_OUT_CHANNELS,C5_OUT_CHANNELS=48,128,197,197,128
C1_STRIDES,C2_STRIDES,C3_STRIDES,C4_STRIDES,C5_STRIDES=4,1,1,1,1
P1_SIZE,P2_SIZE,P5_SIZE=3,3,3
P1_STRIDE,P2_STRIDE,P5_STRIDE=2,2,2
F6_SIZE,F7_SIZE=4096,4096
F8_SIZE=(int)(y_train.shape[1])


# In[7]:

#convolution kernels and bias
C1_kernel=weights([C1_KERNEL_SIZE,C1_KERNEL_SIZE,IMG_DEPTH, C1_OUT_CHANNELS],0.01,'C1_kernel')
C2_kernel=weights([C2_KERNEL_SIZE,C2_KERNEL_SIZE, C1_OUT_CHANNELS, C2_OUT_CHANNELS],0.01,'C2_kernel')
C3_kernel=weights([C3_KERNEL_SIZE,C3_KERNEL_SIZE, C2_OUT_CHANNELS, C3_OUT_CHANNELS],0.01,'C3_kernel')
C4_kernel=weights([C4_KERNEL_SIZE,C4_KERNEL_SIZE, C3_OUT_CHANNELS, C4_OUT_CHANNELS],0.01,'C4_kernel')
C5_kernel=weights([C5_KERNEL_SIZE,C5_KERNEL_SIZE, C4_OUT_CHANNELS, C5_OUT_CHANNELS],0.01,'C5_kernel')

C1_bias=bias([C1_OUT_CHANNELS], 'C1_bias')
C2_bias=bias([C2_OUT_CHANNELS], 'C2_bias')
C3_bias=bias([C3_OUT_CHANNELS], 'C3_bias')
C4_bias=bias([C4_OUT_CHANNELS], 'C4_bias')
C5_bias=bias([C5_OUT_CHANNELS], 'C5_bias')


# In[8]:

#AlexNet network

#Conv layer 1
C1=convLayer(x_train, C1_kernel, C1_STRIDES, 'SAME')
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
F6_weights=weights([num_P5_out, F6_SIZE],0.01,'F6_weights')
F6_bias=bias([F6_SIZE],'F6_bias')
F6=tf.matmul(P5, F6_weights)+F6_bias
ReLU6=tf.nn.relu(F6)
DROP6=tf.nn.dropout(ReLU6, 0.5)

#Fully connected layer 7
F7_weights=weights([F6_SIZE, F7_SIZE],0.01,'F7_weights')
F7_bias=bias([F7_SIZE],'F7_bias')
F7=tf.matmul(DROP6, F7_weights)+F7_bias
ReLU7=tf.nn.relu(F7)
DROP7=tf.nn.dropout(ReLU7, 0.5)

#Fully connected layer 8
F8_weights=weights([F7_SIZE, F8_SIZE],0.01,'F8_weights')
F8_bias=bias([F8_SIZE],'F8_bias')
logits=tf.matmul(DROP7, F8_weights)+F8_bias


# In[16]:

#cost function and accuracy
#cross_entropy = tf.reduce_mean(tf.reduce_sum(y_train*tf.log(y),1))
cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_train, logits=logits))
print(cross_entropy)
train=tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)
correct = tf.equal(tf.argmax(logits, 1), tf.argmax(y_train, 1))
accuracy= tf.reduce_mean(tf.cast(correct, tf.float32))


# In[10]:

config=tf.ConfigProto(allow_soft_placement= True,log_device_placement= True)
sess=tf.Session(config=config)
sess.run(tf.global_variables_initializer())


# In[ ]:

#training
#start=time.time()
for step in range(351,401):
    _, loss, acc = sess.run([train, cross_entropy, accuracy])
    if(step%10==0):
        print (step, loss, acc)
#end=time.time()


# In[14]:

#Save model for continuous learning
saver=tf.train.Saver()
saver.save(sess, 'AlexNet-largeTrain', global_step=350)


# In[15]:

sess.close()


# In[ ]:




# In[ ]:



