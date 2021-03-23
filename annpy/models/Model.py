import annpy
import numpy as np
from annpy.losses.Loss import Loss

class Model():

	def __init__(self, input_shape, input_layer, name):

		self.name = name
		self.input_shape = input_shape
		self.weights = None
		self.loss = None
		self.optimizer = None

	# @abstractmethod
	# def __str__(self):
	# 	return "Model"

	def compile(self, loss, optimizer, metrics):
		# self.loss = annpy.utils.parse.parse_object(loss, annpy.losses.Loss.Loss)
		self.loss = annpy.utils.parse.parse_object(loss, Loss)
		self.optimizer = annpy.utils.parse.parse_object(optimizer, annpy.optimizers.Optimizer.Optimizer)
		self.metrics = []
		for metric in metrics:
			self.metrics.append(annpy.utils.parse.parse_object(metric, annpy.metrics.Metrics.Metrics))

	def forward(self):
		raise NotImplementedError

	def fit(self):
		raise NotImplementedError

	# def evaluate(self, features, targets):
	# 	raise NotImplementedError

	def summary(self, only_model_summary=True):

		print(f"\n-------------------")
		print(f"Summary of: {self.name}")
		print(f"Input shape:  {self.input_shape}")
		self.loss.summary()
		self.optimizer.summary()
		for metric in self.metrics:
			metric.summary()

		if only_model_summary:
			print(f"-------------------\n")

