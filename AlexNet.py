import util
import tensorflow as tf

def classifier(x):
	#parameters for convolution kernels
	IMG_DEPTH=1
	C1_KERNEL_SIZE, C2_KERNEL_SIZE, C3_KERNEL_SIZE, C4_KERNEL_SIZE, C5_KERNEL_SIZE=11,5,3,3,3
	C1_OUT_CHANNELS,C2_OUT_CHANNELS,C3_OUT_CHANNELS,C4_OUT_CHANNELS,C5_OUT_CHANNELS=96,256,384,384,256
	#C1_OUT_CHANNELS,C2_OUT_CHANNELS,C3_OUT_CHANNELS,C4_OUT_CHANNELS,C5_OUT_CHANNELS=48,128,197,197,128
	C1_STRIDES,C2_STRIDES,C3_STRIDES,C4_STRIDES,C5_STRIDES=1,1,1,1,1
	P1_SIZE,P2_SIZE,P5_SIZE=3,3,3
	P1_STRIDE,P2_STRIDE,P5_STRIDE=2,2,2
	F6_SIZE,F7_SIZE=4096,4096
	F8_SIZE=10

	#convolution kernels and bias
	C1_kernel=util.weights([C1_KERNEL_SIZE,C1_KERNEL_SIZE,IMG_DEPTH, C1_OUT_CHANNELS],0.01,'C1_kernel')
	C2_kernel=util.weights([C2_KERNEL_SIZE,C2_KERNEL_SIZE, C1_OUT_CHANNELS, C2_OUT_CHANNELS],0.01,'C2_kernel')
	C3_kernel=util.weights([C3_KERNEL_SIZE,C3_KERNEL_SIZE, C2_OUT_CHANNELS, C3_OUT_CHANNELS],0.01,'C3_kernel')
	C4_kernel=util.weights([C4_KERNEL_SIZE,C4_KERNEL_SIZE, C3_OUT_CHANNELS, C4_OUT_CHANNELS],0.01,'C4_kernel')
	C5_kernel=util.weights([C5_KERNEL_SIZE,C5_KERNEL_SIZE, C4_OUT_CHANNELS, C5_OUT_CHANNELS],0.01,'C5_kernel')

	C1_bias=util.bias([C1_OUT_CHANNELS], 'C1_bias')
	C2_bias=util.bias([C2_OUT_CHANNELS], 'C2_bias')
	C3_bias=util.bias([C3_OUT_CHANNELS], 'C3_bias')
	C4_bias=util.bias([C4_OUT_CHANNELS], 'C4_bias')
	C5_bias=util.bias([C5_OUT_CHANNELS], 'C5_bias')

	#AlexNet network

	#Conv layer 1
	C1=util.convLayer(x, C1_kernel, C1_STRIDES, 'SAME')
	ReLU1=tf.nn.relu(C1+C1_bias)
	P1=util.max_pool(ReLU1,P1_SIZE, P1_STRIDE)
	NORM1=tf.nn.local_response_normalization(P1)

	#Conv layer 2
	C2=util.convLayer(NORM1, C2_kernel, C2_STRIDES, 'SAME')
	ReLU2=tf.nn.relu(C2+C2_bias)
	P2=util.max_pool(ReLU2,P2_SIZE, P2_STRIDE)
	NORM2=tf.nn.local_response_normalization(P2)

	#Conv layer 3
	C3=util.convLayer(NORM2, C3_kernel, C3_STRIDES, 'SAME')
	ReLU3=tf.nn.relu(C3+C3_bias)

	#Conv layer 4
	C4=util.convLayer(ReLU3, C4_kernel, C4_STRIDES, 'SAME')
	ReLU4=tf.nn.relu(C4+C4_bias)

	#Conv layer 5
	C5=util.convLayer(ReLU4, C5_kernel, C5_STRIDES, 'SAME')
	ReLU5=tf.nn.relu(C5+C5_bias)
	P5_pre=util.max_pool(ReLU5,P5_SIZE, P5_STRIDE)

	num_P5_out=(int)(P5_pre.shape[1]*P5_pre.shape[2]*P5_pre.shape[3])
	P5=tf.reshape(P5_pre,[-1,num_P5_out])

	#Fully connected layer 6
	F6_weights=util.weights([num_P5_out, F6_SIZE],0.01,'F6_weights')
	F6_bias=util.bias([F6_SIZE],'F6_bias')
	F6=tf.matmul(P5, F6_weights)
	ReLU6=tf.nn.relu(F6+F6_bias)
	DROP6=tf.nn.dropout(ReLU6, 0.5)

	#Fully connected layer 7
	F7_weights=util.weights([F6_SIZE, F7_SIZE],0.01,'F7_weights')
	F7_bias=util.bias([F7_SIZE],'F7_bias')
	F7=tf.matmul(DROP6, F7_weights)
	ReLU7=tf.nn.relu(F7+F7_bias)
	DROP7=tf.nn.dropout(ReLU7, 0.5)

	#Fully connected layer 8
	F8_weights=util.weights([F7_SIZE, F8_SIZE],0.01,'F8_weights')
	F8_bias=util.bias([F8_SIZE],'F8_bias')
	logits=tf.matmul(DROP7, F8_weights)+F8_bias

	return logits