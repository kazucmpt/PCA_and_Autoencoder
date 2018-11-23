import math
import random
import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from chainer import Chain, Variable,optimizers,iterators,training,dataset
import chainer.links as L 
import chainer.functions as F 
from chainer.training import extensions
from chainer.cuda import to_cpu

class Autoencoder(Chain):
	def __init__(self):
		super().__init__()
		with self.init_scope():
			self.l1 = L.Linear(3,2)
			self.l2 = L.Linear(2,3)

	def __call__(self,x):
		h1 = self.l1(x)
		h2 = self.l2(h1)

		return h1, h2


class TrainWrapper(Chain):
	def __init__(self,model):
		super().__init__()
		with self.init_scope():
			self.model = model

	def __call__(self,x):
		_, y = self.model(x)
		loss = F.mean_squared_error(x,y)

		return loss

class Dataset(dataset.DatasetMixin):
	def __init__(self,number_of_data, show_initial = False):

		noise_level = 0.1

		self.data = np.zeros((number_of_data,3),dtype = np.float32)

		OA_vector = np.array([3,2,0])
		OB_vector = np.array([2,-1,0])

		t = np.random.uniform(-0.5,0.5,number_of_data)
		s = np.random.uniform(-0.5,0.5,number_of_data)

		for i in range(0,number_of_data):
			#noise = np.random.normal(0, noise_level, 3)
			noise = np.random.uniform(-noise_level, noise_level,3)
			self.data[i] = t[i]*OA_vector + s[i]*OB_vector + noise

	def __len__(self):
		return self.data.shape[0]

	def get_example(self,idx):
		return self.data[idx]

	def get_all_data(self):
		return self.data

	def plot_data(self):

		fig = plt.figure()
		ax = Axes3D(fig)

		ax.scatter(self.data[:,0],self.data[:,1],self.data[:,2])
		ax.set_xlim(-3,3)
		ax.set_ylim(-3,3)
		ax.set_zlim(-3,3)
		ax.set_xlabel("x",fontsize=16)
		ax.set_ylabel("y",fontsize=16)
		ax.set_zlabel("z",fontsize=16)
		plt.show()


if __name__ == "__main__":

	n_epoch = 5
	batch_size = 10

	number_of_data = 1000
	train_data = Dataset(number_of_data,False)
	train_data.plot_data()

	# NNのモデル宣言
	model = Autoencoder()
	
	# chainerのoptimizer
	optimizer = optimizers.SGD(lr=0.05).setup(TrainWrapper(model))
	train_iter = iterators.SerialIterator(train_data,batch_size)

	updater = training.StandardUpdater(train_iter,optimizer,device=0)
	trainer = training.Trainer(updater,(n_epoch,"epoch"),out="result")

	trainer.run()

	input_data = train_data.get_all_data()
	input_data = Variable(input_data)
	input_data.to_gpu()

	y, _ = model(input_data)

	y.to_cpu()
	y = y.array #convert to numpy from variable

	plt.scatter(y[:,0],y[:,1])
	plt.show()
