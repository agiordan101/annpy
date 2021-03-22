import annpy
import numpy as np

class Model():

	def __init__(self,
					input_shape=None,
					input_layer=None,
					name="Default models name"):

		self.name = name
		self.input_shape = input_shape
		self.weights = None
		self.loss = None
		self.optimizer = None

	# @abstractmethod
	# def __str__(self):
	# 	return "Model"

	def compile(self,
				loss="MSE",
				optimizer="SGD"):
		self.loss = annpy.utils.parse.parse_object(loss, annpy.losses.Loss)
		self.optimizer = annpy.utils.parse.parse_object(optimizer, annpy.optimizers.Optimizer)

	def forward(self, inputs):
        raise NotImplementedError

	def fit(self,
			features,
			targets,
			validation_features=None,
			validation_targets=None,
			k_fold_as_validation=False,
			k_fold_percent=0.2,
			verbose=False):
        raise NotImplementedError

	def summary(self, only_model_summary=True):

		print(f"\n-------------------")
		print(f"Summary of: {self.name}")
		print(f"Input shape:  {self.input_shape}")
		print(f"Loss: {self.loss}")
		print(f"Optimizer: {self.optimizer}")

		if only_model_summary:
			print(f"-------------------\n")

