from three_layer_neural_network import NeuralNetwork as ThreeNN
import numpy as np

class DeepNeuralNetwork(ThreeNN):
    """
    This class builds and trains a multi-layer neural network
    """
    def __init__(self, nn_input_dim, nn_hidden_dim_list, nn_output_dim, actFun_type='tanh', reg_lambda=0.01, seed=0):
        pass

    def feedforward(self, X, actFun):
        return super().feedforward(X, actFun)
    
    def backprop(self, X, y, actFun):
        return super().backprop(X, y, actFun)
    
    def calculate_loss(self, X, y):
        return super().calculate_loss(X, y)



class Layer(ThreeNN):
    def __init__(self, nn_dim, nn_next_dim, actFun_type='tanh', reg_lambda=0.01, seed=0):
        self.nn_dim = nn_dim
        self.nn_next_dim = nn_next_dim
        self.actFun_type = actFun_type
        self.reg_lambda = reg_lambda

        np.random.seed(seed)
        self.W = np.random.randn(self.nn_dim, self.nn_next_dim) / np.sqrt(self.nn_dim)
        self.b = np.zeros((1, self.nn_next_dim))
    
    def feedforward(self, X, actFun):
        self.z = np.dot(X, self.W) + self.b
        self.a = self.actFun(self.z, self.actFun_type)