import annpy
import numpy as np
from annpy.losses.Loss import Loss
from annpy.metrics.Metric import Metric
from annpy.metrics.Accuracy import Accuracy
from annpy.optimizers.Optimizer import Optimizer

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
		self.optimizer = annpy.utils.parse.parse_object(optimizer, Optimizer)

		# Add loss to metrics
		self.metrics = [self.loss]

		# Save in accuracy_metric the first metric to inherite from Accuracy
		self.accuracy_metric = None
		for metric in metrics:
			self.metrics.append(annpy.utils.parse.parse_object(metric, Metric)) # Autoriser les string pour les metrics ?? Compatibilite dans le fit / parse ?

			if not self.accuracy_metric and issubclass(type(self.metrics[-1]), Accuracy):
				self.accuracy_metric = self.metrics[-1]

		if not self.accuracy_metric:
			self.accuracy_metric = Accuracy()

	def forward(self):
		raise NotImplementedError
	
	def evaluate(self, model, features, targets, verbose=True):

		predictions = model.forward(features)

		self.loss.reset()
		self.loss(predictions, targets)
		loss = self.loss.get_result()

		self.accuracy_metric.reset() # useless
		self.accuracy_metric(predictions, targets)
		accuracy = self.accuracy_metric.get_result()

		print(f"Model evaluation -- loss: {loss} -- accuracy: {accuracy}")
		return loss, accuracy

	def fit(self,
			train_features,
			train_targets,
			batch_size,
			epochs,
			metrics,
			validation_features,
			validation_targets,
			verbose):
		
		if validation_features is None:
			self.validation_features = train_features
			self.validation_targets = train_targets

		# Dataset length
		self.features_len = len(train_features)

		# Batchs number
		self.n_batch = self.features_len // batch_size + (1 if len(train_features) % batch_size else 0)

		# Split dataset into <n_batch> batch of len <batch_size>
		self.batch_split = list(range(0, self.features_len, batch_size))[1:]

	# def get_metrics_data(self):
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

