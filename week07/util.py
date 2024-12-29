import numpy as np

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def turunan_sigmoid(x):
	return sigmoid(x) + (1 - sigmoid(x))

def mse_loss(y_true, y_pred):
	return ((y_true - y_pred) ** 2).mean()

def turunan_mse_loss(y_true, y_pred):
	return -2 * (y_true - y_pred)

def scaling(x):
	return (x - np.min(x)) / (np.max(x) - np.min(x))