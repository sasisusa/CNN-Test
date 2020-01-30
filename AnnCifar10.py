import sys
import time
import argparse
import numpy as np
import tensorflow as tf


#========================================
# input: 32x32x3 (RGB-image)
#========================================
# 3x3 conv, batch-normalization, 90 ReLU
# maxpool 2x2 stride [2,2]
# dropout
#----------------------------------------
# 3x3 conv, batch-normalization, 100 ReLU
# maxpool 2x2 stride [2,2]
# dropout
#----------------------------------------
# 3x3 conv, batch-normalization, 110 ReLU
# maxpool 2x2 stride [2,2]
# dropout
#----------------------------------------
# flatten
#----------------------------------------
# 800 sigmoid
# dropout
#----------------------------------------
# 10 softmax
#========================================
# output: 10
#========================================
class AnnCifar10:
	## depth/size for conv-layers
	c_depth = [90, 100, 110]
	## size for hidden layer
	hid_n = [800]
	def __init__(self, ann_graph=tf.get_default_graph(), seed_init=84561, weights_file="", batchsize=None):
		width = 32
		height = 32
		num_ch = 3
		num_classes = 10
		hid_n = self.hid_n
		c_depth = self.c_depth
		## neurons for 1.fc-layer: 3 pooling layer (2x2,stride [2,2]) => dim/2/2/2 = dim/(2**3)
		fc_first = (width/2/2/2)*(height/2/2/2)*c_depth[-1] 
		with ann_graph.as_default():
			np.random.seed(seed_init)
			tf.set_random_seed(seed_init)
			with tf.name_scope("weights"):
				if weights_file:
					w_load = np.load(weights_file)
					w_load = w_load[()]
					W = {}
					for key in w_load:
						W[key] = tf.Variable(w_load[key], name=key)
				else:
					init_c = tf.contrib.layers.xavier_initializer_conv2d()
					init_bc = tf.zeros_initializer()
					init_w = tf.contrib.layers.xavier_initializer()
					init_b = tf.zeros_initializer()
					W = {
						'C1' : tf.get_variable("C1", shape=[3, 3, num_ch, c_depth[0]], initializer=init_c),
						'BC1' : tf.get_variable("BC1", shape=[c_depth[0]], initializer=init_bc),
						'C2' : tf.get_variable("C2", shape=[3, 3, c_depth[0], c_depth[1]], initializer=init_c),
						'BC2' : tf.get_variable("BC2", shape=[c_depth[1]], initializer=init_bc),
						'C3' : tf.get_variable("C3", shape=[3, 3, c_depth[1], c_depth[2]], initializer=init_c),
						'BC3' : tf.get_variable("BC3", shape=[c_depth[2]], initializer=init_bc),
						'W1' : tf.get_variable("W1", shape=[fc_first, hid_n[0]], initializer=init_w),
						'B1' : tf.get_variable("B1", shape=[hid_n[0]], initializer=init_b),
						'W2' : tf.get_variable("W2", shape=[hid_n[0], num_classes], initializer=init_w),
						'B2' : tf.get_variable("B2", shape=[num_classes], initializer=init_b),
					}
				## variables for batch-normalization
				OffsetCL1 = tf.Variable(tf.zeros([c_depth[0]]))
				ScaleCL1 = tf.Variable(tf.ones([c_depth[0]]))
				OffsetCL2 = tf.Variable(tf.zeros([c_depth[1]]))
				ScaleCL2 = tf.Variable(tf.ones([c_depth[1]]))
				OffsetCL3 = tf.Variable(tf.zeros([c_depth[2]]))
				ScaleCL3 = tf.Variable(tf.ones([c_depth[2]]))
			with tf.name_scope("placeholder"):
				X = tf.placeholder(tf.float32, [batchsize, width, height, num_ch], name='X')
				Y = tf.placeholder(tf.float32, [batchsize, num_classes], name='Y')
				Dropout = tf.placeholder(tf.float32, shape=[], name="dropout")
			def _ann_net(X):
				NormEps = 1e-6
				## convolutional layer 1 / input
				CL1 = tf.nn.bias_add(tf.nn.conv2d(X, W['C1'], strides=[1, 1, 1, 1], padding='SAME'), W['BC1'])
				BatchMeanCL1, BatchVarCL1 = tf.nn.moments(CL1, [0,1,2])
				CL1 = tf.nn.batch_normalization(CL1, BatchMeanCL1, BatchVarCL1, OffsetCL1, ScaleCL1, NormEps)
				CL1 = tf.nn.relu(CL1)
				CL1 = tf.nn.max_pool(CL1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME')
				CL1 = tf.nn.dropout(CL1, keep_prob=(1.0 - 0.5*Dropout))
				## convolutional layer 2
				CL2 = tf.nn.bias_add(tf.nn.conv2d(CL1, W['C2'], strides=[1, 1, 1, 1], padding='SAME'), W['BC2'])
				BatchMeanCL2, BatchVarCL2 = tf.nn.moments(CL2, [0,1,2])
				CL2 = tf.nn.batch_normalization(CL2, BatchMeanCL2, BatchVarCL2, OffsetCL2, ScaleCL2, NormEps)
				CL2 = tf.nn.relu(CL2)
				CL2 = tf.nn.max_pool(CL2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME')
				CL2 = tf.nn.dropout(CL2, keep_prob=(1.0 - 0.5*Dropout))
				## convolutional layer 3
				CL3 = tf.nn.bias_add(tf.nn.conv2d(CL2, W['C3'], strides=[1, 1, 1, 1], padding='SAME'), W['BC3'])
				BatchMeanCL3, BatchVarCL3 = tf.nn.moments(CL3, [0,1,2])
				CL3 = tf.nn.batch_normalization(CL3, BatchMeanCL3, BatchVarCL3, OffsetCL3, ScaleCL3, NormEps)
				CL3 = tf.nn.relu(CL3)
				CL3 = tf.nn.max_pool(CL3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME')
				CL3 = tf.nn.dropout(CL3, keep_prob=(1.0 - 0.5*Dropout))
				## fully-connected layer 1 / flatten / input-layer for fully-connected ANN
				Z1 = tf.contrib.layers.flatten(CL3)
				## fully-connected layer 2 / hidden-layer
				U2 = tf.nn.bias_add(tf.matmul(Z1, W['W1']), W['B1'])
				Z2 = tf.nn.sigmoid(U2)
				Z2 = tf.nn.dropout(Z2, keep_prob=(1.0 - 1.0*Dropout))
				## fully-connected layer 3 / output-layer
				U3 = tf.nn.bias_add(tf.matmul(Z2, W['W2']), W['B2'])
				Z3 = tf.nn.softmax(U3)
				return U3, Z3
			with tf.name_scope("model"):
				U_last, Z_last = _ann_net(X)
			with tf.name_scope("loss"):
				Loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=U_last, labels=Y))
			with tf.name_scope("optimizer"):
				TrainOp = tf.train.AdamOptimizer(epsilon=1e-5).minimize(Loss)
			with tf.name_scope("accuracy"):
				Z_last_lbidx = tf.argmax(Z_last, axis=1)
				Y_lbidx = tf.argmax(Y, axis=1)
				WrongPred = tf.not_equal(Z_last_lbidx, Y_lbidx)
				WrongSum = tf.reduce_sum(tf.cast(WrongPred, tf.float32))
			self.W = W 
			self.X, self.Y = X, Y
			self.TrainOp = TrainOp
			self.Loss, self.WrongSum = Loss, WrongSum
			self.Dropout = Dropout
	def save_weights(self, sess, weights_file='weights_AnnCifar10.npy'):
		w_save = sess.run(self.W)
		np.save(weights_file, w_save)
	def load_weights(self, sess, weights_file='weights_AnnCifar10.npy'):
		w_load = np.load(weights_file)
		w_load = w_load[()]
		for key in w_load:
			self.W[key].load(w_load[key], sess)


def GetCIFAR10_wVal(val_len=0, print_info=True):
	(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
	num_classes = (y_train.max()-y_train.min()+1).astype('int')
	## one-hot vector
	y_train = tf.keras.utils.to_categorical(y_train, num_classes).astype('float32')
	y_test = tf.keras.utils.to_categorical(y_test, num_classes).astype('float32')
	## convert features to values between [0.0, 1.0]
	x_train = x_train.astype('float32')/255.0
	x_test = x_test.astype('float32')/255.0
	## cut out validation data
	(x_val, y_val) =  (x_train[x_train.shape[0]-val_len:x_train.shape[0]], y_train[y_train.shape[0]-val_len:y_train.shape[0]])
	(x_train, y_train) =  (x_train[0:x_train.shape[0]-val_len], y_train[0:y_train.shape[0]-val_len])

	if print_info:
		print('CIFAR10-Dataset with validation-set:')
		print('x_train dtype:', x_train.dtype)
		print('x_train shape:', x_train.shape)
		print('y_train dtype:', y_train.dtype)
		print('y_train shape:', y_train.shape)
		print('relative frequency y_train: ' + ', '.join(map(str, (y_train.sum(axis=0)/float(y_train.shape[0])))))

		print('x_val dtype:', x_val.dtype)
		print('x_val shape:', x_val.shape)
		print('y_val dtype:', y_val.dtype)
		print('y_val shape:', y_val.shape)
		print('relative frequency y_val: ' + ', '.join(map(str, (y_val.sum(axis=0)/float(y_val.shape[0])))))

		print('x_test dtype:', x_test.dtype)
		print('x_test shape:', x_test.shape)
		print('y_test dtype:', y_test.dtype)
		print('y_test shape:', y_test.shape)
		print('relative frequency y_test: ' + ', '.join(map(str, (y_test.sum(axis=0)/float(y_test.shape[0])))))

	return (x_train, y_train), (x_val, y_val), (x_test, y_test)



def AccuracyTest(sess, net, x, y, Batchsize, print_test=True):
	wrong_sum = 0
	loss_test = 0.0
	for batch in range(int(y.shape[0]/Batchsize)):
		BatchX, BatchY = x[batch*Batchsize:(batch+1)*Batchsize], y[batch*Batchsize:(batch+1)*Batchsize]
		lo, ws = sess.run((net.Loss, net.WrongSum), feed_dict={net.X: BatchX, net.Y: BatchY, net.Dropout: 0.0})
		wrong_sum += ws
		loss_test += lo
	loss_test /= int(y.shape[0]/Batchsize)

	if print_test:
		sys.stdout.write("%d\t%f\t%f\t" % (wrong_sum, wrong_sum/(int(y.shape[0]/Batchsize)*Batchsize), loss_test))

	return wrong_sum, loss_test



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--batchsize', action='store', type=int, default=100)
	parser.add_argument('--epochs', action='store', type=int, default=50)
	parser.add_argument('--seed', action='store', type=int, default=84561)
	parser.add_argument('--threads', action='store', type=int, default=4)
	parser.add_argument('--loadweights', action='store_true')
	parser.add_argument('--saveweights', action='store_true')

	args = parser.parse_args()
	Batchsize = args.batchsize
	TrainEpochs = args.epochs
	Threads = args.threads
	seed_init = args.seed
	LoadWeights = args.loadweights
	SaveWeights = args.saveweights

	dropout = 0.5

	ann_graph = tf.get_default_graph()
	net = AnnCifar10(ann_graph, seed_init=seed_init)

	(x_train, y_train), (x_val, y_val), (x_test, y_test) = GetCIFAR10_wVal(val_len=10000)

	with tf.Session(graph=ann_graph, config=tf.ConfigProto(intra_op_parallelism_threads=Threads)) as sess:
		sess.run(tf.global_variables_initializer())
		if LoadWeights:
			net.load_weights(sess) ## load already stored the weights from previous or existing 
		print('\nBatchsize: %d\nTrain-epochs: %d\nThreads: %d\nSeed: %d\n' % (Batchsize, TrainEpochs, Threads, seed_init))
		print('Epochsizes:')
		sys.stdout.write('Train: %d, ' % (int(y_train.shape[0]/Batchsize)*Batchsize))
		sys.stdout.write('Val: %d, ' % (int(y_val.shape[0]/Batchsize)*Batchsize))
		print('Test %d\n' % (int(y_test.shape[0]/Batchsize)*Batchsize))

		MinWrongTest = x_test.shape[0]
		MinEpochTest = 0
		MinWrongVal = x_val.shape[0]
		MinEpochVal = 0
		WrongTestAtBestVal = x_test.shape[0]
		NewBest = False

		print('### Begin: Table ###')
		sys.stdout.write("Epoch\tWrongTrain\tErrorTrain\tLossTrain\tWrongVal\tErrorVal\t") 
		sys.stdout.write("LossVal\tWrongTest\tErrorTest\tLossTest\tTimer\n") 

		StartTime = time.time()
		epoch = 0;
		while epoch < TrainEpochs:
			EpochStartTime = time.time()
			## trainings-data shuffle
			rnd_idx = np.random.permutation(len(x_train))
			x, y = x_train[rnd_idx], y_train[rnd_idx]

			## mini-batch training (weight optimization)
			for batch in range(int(x_train.shape[0]/Batchsize)):
				BatchX, BatchY = x[batch*Batchsize:(batch+1)*Batchsize], y[batch*Batchsize:(batch+1)*Batchsize]
				sess.run(net.TrainOp, feed_dict={net.X: BatchX, net.Y: BatchY, net.Dropout: dropout})
			epoch += 1

			sys.stdout.write("%d\t" % (epoch))
			AccuracyTest(sess, net, x_train, y_train, Batchsize)
			wrong_sum, loss_test = AccuracyTest(sess, net, x_val, y_val, Batchsize)
			if wrong_sum < MinWrongVal:
				NewBest = True
				MinWrongVal = wrong_sum
				MinEpochVal = epoch
				if SaveWeights:
					net.save_weights(sess) ## save the weights from the best validation-test
			## show test-accuracy (here/demo: non-practical scenario)
			wrong_sum, loss_test = AccuracyTest(sess, net, x_test, y_test, Batchsize)
			if NewBest:
				NewBest = False
				WrongTestAtBestVal = wrong_sum
			if wrong_sum < MinWrongTest:
				MinWrongTest = wrong_sum
				MinEpochTest = epoch

			sys.stdout.write("%.2f\n" % (time.time() - StartTime))
			sys.stdout.flush()

		print('### End: Table ###')
		print('Best-Val: %d, Epoch: %d' % (MinWrongVal, MinEpochVal))
		print('TestAtBestVal: %d, Epoch: %d' % (WrongTestAtBestVal, MinEpochVal))
		print('Best-Test: %d, Epoch: %d' % (MinWrongTest, MinEpochTest))




