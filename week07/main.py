import numpy as np
from util import *

class JaringanSyaraf:
	def __init__(self, learning_rate=0.01, epochs = 1000):

		self.w1 = np.random.normal()
		self.w2 = np.random.normal()
		self.w3 = np.random.normal()
		self.w4 = np.random.normal()
		self.w5 = np.random.normal()
		self.w6 = np.random.normal()

		self.b1 = np.random.normal()
		self.b2 = np.random.normal()
		self.b3 = np.random.normal()

		self.learning_rate = learning_rate
		self.epochs = epochs

	def feedforward(self, inputan):

		x1, x2 = inputan
		
		h1 = sigmoid(x1 * self.w1 + x2 * self.w3 + self.b1)
		h2 = sigmoid(x1 * self.w2 + x2 * self.w4 + self.b2)

		o1 = sigmoid(h1 * self.w5 + h2 * self.w6 + self.b3)

		return o1

	def train(self, X, labels):
		return 1
		# for epoch in range(epochs):
		# 	for x, y in zip(X, labels):


if __name__ == '__main__':
	inputan = np.array([
		[170, 80, 1],
		[172, 81, 1],
		[160, 64, 0],
		[155, 59, 0],
		[167, 74, 1],
		[156, 57, 0]
	], dtype=float)


	x1 = inputan[:,0]
	x2 = inputan[:,1]
	scaled_x1 = scaling(x1)
	scaled_x2 = scaling(x2)

	inputan[:, 0] = scaled_x1
	inputan[:, 1] = scaled_x2
	