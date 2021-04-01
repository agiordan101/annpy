import numpy as np
from annpy.metrics.Metric import Metric

class Accuracy(Metric):
	
	def __init__(self):
		super().__init__()
		self.name = "Accuracy"

	# Depend on child class
	def accuracy_conditions(self, prediction, target):
		return np.array_equal(prediction, target)

	# Depend on child class
	def compute(self, predictions, targets):
		return len([1 for prediction, target in zip(predictions, targets) if self.accuracy_conditions(prediction, target)]) / len(predictions)
		# return len([1 for prediction, target in zip(predictions, targets) if self.accuracy_conditions(prediction, target)])

	# def get_mem_len_append(self, predictions, targets):
	# 	return len(predictions)

	def summary(self):
		print(f"Metric:\t\tannpy.accuracies.Accuracy, (Exact values tested)")

