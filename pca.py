import numpy as np 
import random
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def generate_dataset(number_of_data, show_initial = True):

	noise_level = 0.1

	data = np.zeros((number_of_data,3))

	OA_vector = np.array([3,2,0])
	OB_vector = np.array([2,-1,0])

	t = np.random.uniform(-0.5,0.5,number_of_data)
	s = np.random.uniform(-0.5,0.5,number_of_data)

	for i in range(0,number_of_data):
		noise = np.random.uniform(-noise_level, noise_level,3)
		data[i] = t[i]*OA_vector + s[i]*OB_vector + noise


	if show_initial:

		fig = plt.figure()
		ax = Axes3D(fig)
	
		ax.scatter(data[:,0],data[:,1],data[:,2])
		ax.set_xlim(-3,3)
		ax.set_ylim(-3,3)
		ax.set_zlim(-3,3)
		ax.set_xlabel("x",fontsize=16)
		ax.set_ylabel("y",fontsize=16)
		ax.set_zlabel("z",fontsize=16)
		plt.show()

	return data.T

def generate_normalized_dataset(number_of_data,data):

	average_of_each_dimension = np.array([np.average(data[0,:]),np.average(data[1,:]),np.average(data[2,:])])

	for i in range(3):
		for j in range(number_of_data):
			data[i][j] = data[i][j] - average_of_each_dimension[i]

	return data

if __name__ == "__main__":

	number_of_data = 1000
	
	data = generate_dataset(number_of_data,show_initial = True)

	normalized_data = generate_normalized_dataset(number_of_data,data)
	variance_covariance_matrix = 1/number_of_data*np.dot(normalized_data,normalized_data.T)

	_,_,V = np.linalg.svd(variance_covariance_matrix,full_matrices=2)
	projected_data = np.dot(V[:2],data)

	plt.scatter(projected_data[0],projected_data[1])
	plt.show()
