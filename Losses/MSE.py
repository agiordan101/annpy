import numpy as np

# class MSE():

# 	def __init__(self):

# 	def forward(self, weights, deriv):
# 		return weights - self.lr * deriv

# 	# def backward(self):
# 	# 	pass

def MSE(prediction, target):
	diff = target - prediction
	return np.mean(diff * diff)
