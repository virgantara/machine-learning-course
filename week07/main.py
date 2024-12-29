import numpy as np
from util import *
import matplotlib.pyplot as plt

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

	def feedforward(self, X):

		x1, x2 = X[0], X[1]
		
		h1 = sigmoid(self.w1 * x1 + self.w2 * x2 + self.b1)
		h2 = sigmoid(self.w3 * x1 + self.w4 * x2 + self.b2)
		o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)

		return o1

	def train(self, X, labels):

		losses = []
		for epoch in range(self.epochs):
			for x, y_true in zip(X, labels):
				x1, x2 = x[0], x[1]

				h1 = x1 * self.w1 + x2 * self.w2 + self.b1
				h2 = x1 * self.w3 + x2 * self.w4 + self.b2

				out_h1 = sigmoid(h1)
				out_h2 = sigmoid(h2)

				o1 = out_h1 * self.w5 + out_h2 * self.w6 + self.b3

				out_o1 = sigmoid(o1)

				y_pred = out_o1

				d_L_d_yp = turunan_mse_loss(y_true, y_pred)
				
				d_yp_d_w5 = h1 * turunan_sigmoid(o1)
				d_yp_d_w6 = h2 * turunan_sigmoid(o1)
				d_yp_d_b3 = turunan_sigmoid(o1)

				d_yp_d_h1 = self.w5 * turunan_sigmoid(o1)
				d_yp_d_h2 = self.w6 * turunan_sigmoid(o1)
				
				d_h1_d_w1 = x1 * turunan_sigmoid(h1)
				d_h1_d_w2 = x2 * turunan_sigmoid(h1)
				d_h1_d_b1 = turunan_sigmoid(h1)
				
				d_h2_d_w3 = x1 * turunan_sigmoid(h2)
				d_h2_d_w4 = x2 * turunan_sigmoid(h2)
				d_h2_d_b2 = turunan_sigmoid(h2)

				d_grad_w1 = d_L_d_yp * d_yp_d_h1 * d_h1_d_w1
				d_grad_w2 = d_L_d_yp * d_yp_d_h1 * d_h1_d_w2
				d_grad_b1 = d_L_d_yp * d_yp_d_h1 * d_h1_d_b1

				d_grad_w3 = d_L_d_yp * d_yp_d_h2 * d_h2_d_w3
				d_grad_w4 = d_L_d_yp * d_yp_d_h2 * d_h2_d_w4
				d_grad_b2 = d_L_d_yp * d_yp_d_h2 * d_h2_d_b2
				
				d_grad_w5 = d_L_d_yp * d_yp_d_w5
				d_grad_w6 = d_L_d_yp * d_yp_d_w6
				d_grad_b3 = d_L_d_yp * d_yp_d_b3

				self.w1 = self.w1 - self.learning_rate * d_grad_w1
				self.w2 = self.w2 - self.learning_rate * d_grad_w2
				self.b1 = self.b1 - self.learning_rate * d_grad_b1

				self.w3 = self.w3 - self.learning_rate * d_grad_w3
				self.w4 = self.w4 - self.learning_rate * d_grad_w4
				self.b2 = self.b2 - self.learning_rate * d_grad_b2

				self.w5 = self.w5 - self.learning_rate * d_grad_w5
				self.w6 = self.w6 - self.learning_rate * d_grad_w6
				self.b3 = self.b3 - self.learning_rate * d_grad_b3
				

			if epoch % 10 == 0:
				y_preds = np.apply_along_axis(self.feedforward,1, X)
				loss = mse_loss(labels, y_preds)
				losses.append(loss)
				print("Epoch %d loss: %.3f" % (epoch, loss))

		
		return losses


if __name__ == '__main__':
	data = np.array([
		[170, 80, 1],
		[172, 81, 1],
		[160, 64, 0],
		[155, 59, 0],
		[167, 74, 1],
		[156, 57, 0]
	], dtype=float)

	
	X = data
	
	labels = X[:,2]
	X = X[:,[0,1]]
	X[:, 0] = scaling(X[:,0])
	X[:, 1] = scaling(X[:,1])

	num_epoch = 1000
	lr = 0.1
	jst = JaringanSyaraf(epochs=num_epoch, learning_rate=lr)
	losses = jst.train(X, labels)

	sumbu_x = np.arange(1,101)
	
	plt.plot(sumbu_x, losses)
	plt.show()