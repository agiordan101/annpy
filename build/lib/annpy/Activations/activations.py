import math
import numpy as np

def ReLU(x):
	return np.where(x < 0, 0, x)

def linear(x):
	return x

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
	s = sigmoid(x)
	return s * (1 - s)

Activations = {
	"linear": (linear, linear),
	"ReLU": (ReLU, linear),
	"sigmoid": (sigmoid, sigmoid_deriv)
}
