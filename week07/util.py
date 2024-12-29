import numpy as np

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def turunan_sigmoid(x):
	fx = sigmoid(x)
	return fx * (1 - fx)

def mse_loss(y_true, y_pred):
	return ((y_true - y_pred) ** 2).mean()

def turunan_mse_loss(y_true, y_pred):
	return -2 * (y_true - y_pred)

def scaling(x):
	return (x - np.min(x)) / (np.max(x) - np.min(x))

def std_scaling(x):
	mean = np.mean(x)
	std = np.std(x)
	return (x - mean) / std