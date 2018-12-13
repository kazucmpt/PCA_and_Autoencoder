import math
from math import sin,cos
import numpy as np
import matplotlib.pyplot as plt
import random
import chainer.functions as F
import chainer.links as L
from chainer import Chain, Variable,optimizers
from mpl_toolkits.mplot3d import Axes3D

class Autoencoder(Chain):
	def __init__(self):
		super(Autoencoder,self).__init__(
			l1 = L.Linear(3,2),
			l2 = L.Linear(2,3)
		)

	def __call__(self,x):
		h = F.relu(self.l1(x))
		h = self.l2(h)

		return h

def generate_plane_data(number_of_data, noise_level):
	data = np.zeros((number_of_data, 3))
	OA_vector = np.array([3, 2, 1])
	OB_vector = np.array([2, -1, -2])

	t = np.random.uniform(-0.5, 0.5, number_of_data)
	s = np.random.uniform(-0.5, 0.5, number_of_data)

	for i in range(0,number_of_data):
		noise = np.random.normal(0, noise_level,3)
		data[i] = t[i]*OA_vector + s[i]*OB_vector + noise

	return data

def generate_sphere_data(number_of_data, noise_level):
	data = np.zeros((number_of_data, 3))

	inner_sphere_radius = 3.0
	outer_sphere_radius = 1.0

	for i in range(0,number_of_data):
		if i > int( number_of_data / 2 ):
			r = random.gauss(inner_sphere_radius,noise_level)
		else:
			r = random.gauss(outer_sphere_radius,noise_level)

		theta = random.uniform(0, 2*math.pi)
		phi = random.uniform(0, math.pi)
		data[i] = [r*sin(theta)*cos(phi), r*sin(theta)*sin(phi), r*cos(theta)]

	return data

def show_data(x):
	if x.shape[1] == 2:
		plt.scatter(x[:,0],x[:,1])
		plt.xlim([-3,3])
		ply.ylim([-3,3])
		plt.xlabel("x",fontsize=16)
		plt.ylabel("y",fontsize=16)
		plt.show()

	if x.shape[1] == 3:
		fig = plt.figure()
		ax = Axes3D(fig)
		ax.scatter(x[:,0],x[:,1],x[:,2])
		ax.set_xlim(-3,3)
		ax.set_ylim(-3,3)
		ax.set_zlim(-3,3)
		ax.set_xlabel("x", fontsize=16)
		ax.set_ylabel("y", fontsize=16)
		ax.set_zlabel("z", fontsize=16)
		plt.show()

def main():
	number_of_data = 1500
	noise_level = 0.1
	batchsize = 128
	max_epoch = 120

	x = generate_plane_data(number_of_data,noise_level)
	show_data(x)

	x_train = Variable(np.array(x, dtype=np.float32))

	model = Autoencoder()
	optimizer = optimizers.MomentumSGD(lr=0.01, momentum=0.9).setup(model)

	perm = np.random.permutation(number_of_data)
	for epoch in range(max_epoch):
		for i in range(0,number_of_data,batchsize):
			x_train_batch = x_train[perm[i:i + batchsize]]

			model.cleargrads()
			t = model(x_train_batch)
			loss = F.mean_squared_error(t, x_train_batch)
			loss.backward()
			optimizer.update()
		if epoch % 10 == 0:
			print("epoch:", epoch, "loss", loss.data)

	decoded_x = model(x_train)
	decoded_x = decoded_x.array
	show_data(decoded_x)

	print(x[1])
	print(model(Variable(np.array([x[1]], dtype=np.float32))).array)


if __name__ == "__main__":
	main()
