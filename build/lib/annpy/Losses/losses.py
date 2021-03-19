import numpy as np

def MSE(prediction, target):
	diff = target - prediction
	return np.mean(diff * diff)
