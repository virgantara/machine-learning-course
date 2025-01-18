import numpy as np
from util import *
import matplotlib.pyplot as plt

class JaringanSyaraf:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01, epochs=1000):
        
        self.W1 = np.random.random(size=(hidden_size, input_size))
        self.b1 = np.random.random(size=(hidden_size, 1))

        self.W2 = np.random.random(size=(output_size, hidden_size))
        self.b2 = np.random.random(size=(output_size, 1)) 
       
        self.learning_rate = learning_rate
        self.epochs = epochs

    def feedforward(self, X):


        Z1 = np.matmul(self.W1, X.T) + self.b1
        # print(self.W1.shape, X.T.shape, self.b1.shape)
        A1 = sigmoid(Z1) 

        Z2 = np.matmul(self.W2, A1) + self.b2
        A2 = sigmoid(Z2)

        return Z1, Z2, A1, A2

    def train(self, X, labels):

        losses = []
        for epoch in range(self.epochs):
            
            Z1, Z2, A1, A2 = self.feedforward(X)

            loss = mse_loss(labels, A2)
            losses.append(loss)

            dL_dA2 = turunan_mse_loss(labels, A2).T
            dA2_dZ2 = turunan_sigmoid(Z2)

            dZ2_dW2 = A1

            delta_2 = (dL_dA2 * dA2_dZ2.T)
            dL_dW2 = np.matmul(delta_2.T, dZ2_dW2.T)
            dZ2_db2 = 1
            dL_db2 = delta_2.T * dZ2_db2
            print(dL_db2.shape)
            self.W2 = self.W2 - self.learning_rate * dL_dW2
            
            

            # dL_dW1 = np.dot(dL_dA1 * dA1_dz1, dZ1_db1)
            # dL_db1 = np.dot(dL_dA1 * dA1_dz1, dZ1_db1)
            # 
            # self.W1 = self.W1 - self.learning_rate * dL_dW1
            # self.b1 = self.b1 - self.learning_rate * dL_db1

            # if epoch % 10 == 0:
            #     print(f"Epoch {epoch}, Loss: {loss:.3f}")

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

    X = data[:, :2]
    labels = data[:, 2]

    # Normalize input features
    X[:, 0] = scaling(X[:, 0])
    X[:, 1] = scaling(X[:, 1])
    

    input_size = 2
    hidden_size = 3
    output_size = 1
    learning_rate = 0.1
    epochs = 1

    # Initialize and train neural network
    jst = JaringanSyaraf(input_size, hidden_size, output_size, learning_rate, epochs)
    

    losses = jst.train(X,labels)
    
    # Plot the loss curve
    # plt.plot(range(epochs), losses)
    # plt.xlabel("Epochs")
    # plt.ylabel("Loss")
    # plt.title("Loss Curve")
    # plt.show()
    # losses = jst.train(X, labels)

    # # Plot the loss curve
    # plt.plot(range(epochs), losses)
    # plt.xlabel("Epochs")
    # plt.ylabel("Loss")
    # plt.title("Loss Curve")
    # plt.show()
