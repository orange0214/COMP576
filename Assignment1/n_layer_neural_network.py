from three_layer_neural_network import NeuralNetwork as ThreeNN
import numpy as np

class DeepNeuralNetwork(ThreeNN):
    """
    This class builds and trains a multi-layer neural network
    """
    def __init__(self, nn_dim_list, actFun_type='tanh', reg_lambda=0.01, seed=0):
        self.nn_dim_list = nn_dim_list
        self.actFun_type = actFun_type
        self.reg_lambda = reg_lambda
        self.seed = seed
        self.layers = []
        for i in range(len(self.nn_dim_list) - 1):
            self.layers.append(Layer(self.nn_dim_list[i], self.nn_dim_list[i+1], self.actFun_type, self.reg_lambda, self.seed))
    
    def calculate_loss(self, X, y):
        num_examples = len(X)
        data_loss = -np.sum(np.log(self.probs[range(len(y)), y]))
        # Add regulatization term to loss (optional)
        for layer in self.layers:
            data_loss += self.reg_lambda / 2 * (np.sum(np.square(layer.W)) + np.sum(np.square(layer.b)))
        loss = (1. / num_examples) * data_loss
        return loss

    def fit_model(self, X, y, epsilon=0.01, num_passes=20000, print_loss=True):
        input_X = X
        # Epoch
        for i in range(1):
            # Forward propagation in hidden layers
            for j in range(0, len(self.layers)-1):
                input_X = self.layers[j].feedforward(input_X)
            # Forward propagation in output layer
            self.probs = np.exp(input_X) / np.sum(np.exp(input_X), axis=1, keepdims=True)

            # calculate delta
            delta = self.probs
            delta[range(len(X)), y] -= 1
            # Backpropagation in each layer
            for j in range(len(self.layers)-2, -1, -1):
                dW, db = self.layers[j].backprop(X, y, delta)
                self.layers[j].W += -epsilon * dW
                self.layers[j].b += -epsilon * db
                delta = np.dot(delta, self.layers[j].W.T) * self.layers[j].actFun_diff(self.layers[j].z, self.actFun_type)

            # calculate loss
            if print_loss and i % 1000 == 0:
                loss = self.calculate_loss(X, y)
                print("Loss after iteration %i: %f" % (i, loss))


class Layer():
    """
    This class builds and trains a multi-layer neural network
    Implements the feedforward and backpropagation for a single layer
    """
    def __init__(self, nn_dim, nn_next_dim, actFun_type='tanh', reg_lambda=0.01, seed=0):
        self.nn_dim = nn_dim
        self.nn_next_dim = nn_next_dim
        self.actFun_type = actFun_type
        self.reg_lambda = reg_lambda

        np.random.seed(seed)
        self.W = np.random.randn(self.nn_dim, self.nn_next_dim) / np.sqrt(self.nn_dim)
        self.b = np.zeros((1, self.nn_next_dim))

    def actFun(self, z, actFun_type):
        if actFun_type == 'tanh':
            return np.tanh(z)
        elif actFun_type == 'relu':
            return np.maximum(0, z)
        elif actFun_type == 'sigmoid':
            return 1 / (1 + np.exp(-z))
        else:
            raise ValueError('Invalid activation function type')
        
    def actFun_diff(self, z, actFun_type):
        if actFun_type == 'tanh':
            return 1 - np.tanh(z)**2
        elif actFun_type == 'relu':
            return np.where(z > 0, 1, 0)
        elif actFun_type == 'sigmoid':
            return np.exp(-z) / (1 + np.exp(-z))**2
        else:
            raise ValueError('Invalid activation function type')
    
    def feedforward(self, X):
        self.z = np.dot(X, self.W) + self.b
        self.a = self.actFun(self.z, self.actFun_type)
        return self.a
    
    def backprop(self, X, y, delta):
        dW = np.dot(self.a.T, delta)
        db = np.sum(delta, axis=0, keepdims=True)
        print(dW, db)
        return dW, db


def main():
    nn_dim_list = [2, 3, 3, 3, 2]
    model = DeepNeuralNetwork(nn_dim_list, actFun_type='relu')
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    model.fit_model(X, y)

if __name__ == "__main__":
    main()