import numpy as np
from Neuron import Neuron

class NeuralNetworkku:

	def __init__(self):
		bobot = np.array([0, 1])
		bias = 0

		self.h1 = Neuron(bobot, bias)
		self.h2 = Neuron(bobot, bias)
		self.o1 = Neuron(bobot, bias)


	def feedforward(self, inputan):

		act_h1 = self.h1.feedforward(inputan)
		act_h2 = self.h2.feedforward(inputan)

		inputan_o1 = np.array([act_h1, act_h2])
		act_o1 = self.o1.feedforward(inputan_o1)

		return act_o1


if __name__ == "__main__":

	neuron = NeuralNetworkku()

	X = np.array([2,3])
	print(neuron.feedforward(X))