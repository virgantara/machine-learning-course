import numpy as np
from util import sigmoid

class Neuron:
	def __init__(self, bobot, bias):

		self.bobot = bobot
		self.bias = bias

	def feedforward(self, inputan):
		output = np.dot(self.bobot, inputan) + self.bias

		return sigmoid(output)


if __name__ == "__main__":

	bobot = np.array([0, 1])
	bias  = 1

	neuron = Neuron(bobot, bias)

	X = np.array([4,5])
	print(neuron.feedforward(X))