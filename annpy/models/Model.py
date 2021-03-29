import annpy
import numpy as np
import pandas as pd
import plotly.express as px

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

		# Save in accuracy the first metric to inherite from Accuracy
		self.accuracy = None
		for metric in metrics:
			self.metrics.append(annpy.utils.parse.parse_object(metric, Metric)) # Autoriser les string pour les metrics ?? Compatibilite dans le fit / parse ?

			if not self.accuracy and issubclass(type(self.metrics[-1]), Accuracy):
				self.accuracy = self.metrics[-1]

		if not self.accuracy:
			self.accuracy = Accuracy()

	def forward(self):
		raise NotImplementedError
	
	def evaluate(self, model, features, targets, verbose=True):

		predictions = model.forward(features)

		self.loss.reset(save=False)
		self.loss(predictions, targets)
		loss = self.loss.get_result()

		self.accuracy.reset(save=False)
		self.accuracy(predictions, targets)
		accuracy = self.accuracy.get_result()

		print(f"Model evaluation -- loss: {loss} -- accuracy: {accuracy}")
		return loss, accuracy

	def split_dataset(self, a, b, batch_split):

		# Shuffle
		seed = np.random.get_state()
		np.random.shuffle(a)
		np.random.set_state(seed)
		np.random.shuffle(b)

		# Split batches
		a = np.split(a, batch_split)
		b = np.split(b, batch_split)

		# Merge
		return list(zip(a, b))

	def fit(self,
			train_features,
			train_targets,
			batch_size,
			epochs,
			callbacks,
			valid_features,
			valid_targets,
			valid_percent,
			verbose):

		if valid_features is None:

			# Split dataset in 2 -> New train dataset & validation dataset
			datasets = self.split_dataset(train_features, train_targets, [int(valid_percent * len(train_features))])

			self.valid_features = datasets[0][0]
			self.valid_targets = datasets[0][1]
			self.train_features = datasets[1][0]
			self.train_targets = datasets[1][1]

		# Train dataset length
		self.ds_train_len = len(self.train_features)

		# Batchs number
		self.n_batch = self.ds_train_len // batch_size + (1 if self.ds_train_len % batch_size else 0)

		# Split dataset into <n_batch> batch of len <batch_size>
		self.batch_split = list(range(0, self.ds_train_len, batch_size))[1:]

	# def get_metrics_data(self):
	# 	raise NotImplementedError

	def print_graph(self, metrics=[]):

		if not metrics:
			metrics = self.metrics

		data = {}
		for metric in metrics:
			data[metric.get_obj_name()] = metric.get_mem()
			metric.hard_reset()

		# for k, v in data.items():
		# 	print(f"{k} {len(v)}:\n{v}")

		data_df = pd.DataFrame(data)
		fig = px.line(data_df)
		fig.show()
		# exit(0)

	def summary(self, only_model_summary=True):

		print(f"\n-------------------")
		print(f"Summary of:\t{self.name}")

		self.optimizer.summary()
		for metric in self.metrics:
			metric.summary()

		if only_model_summary:
			print(f"-------------------\n")
