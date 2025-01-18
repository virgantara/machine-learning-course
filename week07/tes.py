import numpy as np
from util import *
import matplotlib.pyplot as plt

class JaringanSyaraf:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01, epochs=1000):
        # Initialize weights and biases with random values
        self.W1 = np.random.normal(size=(hidden_size, input_size))  # Input to hidden weights
        self.b1 = np.random.normal(size=(hidden_size, 1))          # Hidden biases

        self.W2 = np.random.normal(size=(output_size, hidden_size))  # Hidden to output weights
        self.b2 = np.random.normal(size=(output_size, 1))            # Output biases

        self.learning_rate = learning_rate
        self.epochs = epochs

    def feedforward(self, X):
        # Forward pass
        Z1 = np.dot(self.W1, X.T) + self.b1  # Weighted sum at hidden layer
        A1 = sigmoid(Z1)               # Activation at hidden layer

        Z2 = np.dot(self.W2, A1) + self.b2  # Weighted sum at output layer
        A2 = sigmoid(Z2)                   # Activation at output layer

        return A2.T

    def train(self, X, labels):
        labels = labels.reshape(-1, 1)  # Ensure labels are column vector
        losses = []

        for epoch in range(self.epochs):
            # Forward pass
            # A2 = self.feedforward(X)
            Z1 = np.dot(self.W1, X.T) + self.b1
            A1 = sigmoid(Z1)
            Z2 = np.dot(self.W2, A1) + self.b2
            A2 = sigmoid(Z2)
            
            loss = mse_loss(labels, A2.T)
            losses.append(loss)

            # Backward pass
            d_L_d_A2 = turunan_mse_loss(labels, A2.T).T  # Gradient of loss with respect to A2

            d_A2_d_Z2 = turunan_sigmoid(Z2)  # Derivative of sigmoid for output layer
            d_Z2_d_W2 = A1                  # Hidden layer activations
            d_Z2_d_b2 = 1                        # Bias derivative is 1

            d_L_d_W2 = np.dot(d_L_d_A2 * d_A2_d_Z2, d_Z2_d_W2.T)
            d_L_d_b2 = np.sum(d_L_d_A2 * d_A2_d_Z2, axis=1, keepdims=True)

            d_Z2_d_A1 = self.W2.T
            d_L_d_A1 = np.dot(d_Z2_d_A1, d_L_d_A2 * d_A2_d_Z2)

            d_A1_d_Z1 = turunan_sigmoid(Z1)  # Derivative of sigmoid for hidden layer
            d_Z1_d_W1 = X.T                       # Input features
            d_Z1_d_b1 = 1                         # Bias derivative is 1

            d_L_d_W1 = np.dot(d_L_d_A1 * d_A1_d_Z1, d_Z1_d_W1.T)
            d_L_d_b1 = np.sum(d_L_d_A1 * d_A1_d_Z1, axis=1, keepdims=True)

            # Update weights and biases
            self.W1 -= self.learning_rate * d_L_d_W1
            self.b1 -= self.learning_rate * d_L_d_b1
            self.W2 -= self.learning_rate * d_L_d_W2
            self.b2 -= self.learning_rate * d_L_d_b2

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.3f}")

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
    hidden_size = 2
    output_size = 1
    learning_rate = 0.1
    epochs = 1000

    # Initialize and train neural network
    jst = JaringanSyaraf(input_size, hidden_size, output_size, learning_rate, epochs)
    losses = jst.train(X, labels)

    # Plot the loss curve
    plt.plot(range(epochs), losses)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.show()
