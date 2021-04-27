import annpy
import numpy as np
import pandas as pd
import plotly.express as px

# from sklearn.model_selection import train_test_split

from annpy.losses.Loss import Loss
from annpy.metrics.Metric import Metric
from annpy.metrics.Accuracy import Accuracy
from annpy.optimizers.Optimizer import Optimizer


from annpy.losses.BinaryCrossEntropy import BinaryCrossEntropy
from annpy.metrics.RangeAccuracy import RangeAccuracy

from abc import ABCMeta, abstractmethod

class Model(metaclass=ABCMeta):

	debug = []
	# train_features: np.array
	# train_targets: np.array
	# val_features: np.array
	# val_targets: np.array

	@classmethod
	def shuffle_datasets(cls, a, b, copy=False):

		if copy:
			a = a.copy()
			b = b.copy()

		seed = np.random.get_state()
		np.random.shuffle(a)
		np.random.set_state(seed)
		np.random.shuffle(b)
		return a, b
	
	@classmethod
	def train_val_split(cls, features, targets, val_percent, shuffle=True):

		if shuffle:
			features, targets = cls.shuffle_datasets(features, targets)

		i = int(val_percent * len(features))
		return features[i:, :], targets[i:, :], features[:i, :], targets[:i, :]

	@classmethod
	def split_dataset_batches(cls, a, b, n_batch, shuffle=True):

		# Shuffle
		if shuffle:
			a, b = cls.shuffle_datasets(a, b)

		# Split batches
		a = np.array_split(a, n_batch)
		b = np.array_split(b, n_batch)

		# Merge
		return list(zip(a, b))


	def __init__(self, input_shape, input_layer, name):

		self.name = name
		self.input_shape = input_shape
		self.weights = None

		"""
			Metric data storage:
				- val_metrics with all validation/evaluation metrics
				- train_metrics with all metrics use in fit
				- current_metrics with train_metrics or val_metrics/train_metrics
			
			Probleme: Les dict doivent etre toujours cree ? 
						Ou cree seulement lorsque le fit en as besoin ?
		"""

		self.metrics = {}				# All metrics used for all fitting (train & validation)
		self.train_metrics = {}			# Metrics use on train dataset
		self.val_metrics = {}			# Metrics use on valid dataset (tmp var for compile)
		self.current_metrics = {}		# Metrics use on train/valid datasets
		self.eval_metrics = {}			# Metrics use to validate each epoch

		self.loss = None
		self.optimizer = None
		self.accuracy = None

		self.stop_trainning = False
		self.val_on = True

	@abstractmethod
	def __str__(self):
		pass
	
	@abstractmethod
	def forward(self):
		pass

	def add_metric(self, metric):

		cpy = metric.copy().set_name('val_' + str(metric))
		self.metrics[str(cpy)] = cpy
		self.val_metrics[str(cpy)] = cpy

		self.metrics[str(metric)] = metric
		self.train_metrics[str(metric)] = metric


	def compile(self, loss, optimizer, metrics):

		if not isinstance(metrics, list):
			raise Exception("Metrics parameter in Model.compile() is not a list")

		self.optimizer = annpy.utils.parse.parse_object(optimizer, Optimizer)

		# Add loss to metrics
		self.loss = annpy.utils.parse.parse_object(loss, Loss)
		self.add_metric(self.loss)
		# self.loss.append_into(self.metrics)

		for metric in metrics:
			metric = annpy.utils.parse.parse_object(metric, Metric)
			self.add_metric(metric)
			# metric.append_into(self.metrics)

			# Save in accuracy the first metric to inherite from Accuracy
			if not self.accuracy and issubclass(type(metric), Accuracy):
				self.accuracy = metric

		if not self.accuracy:
			self.accuracy = Accuracy()
			self.add_metric(self.accuracy)
			# self.accuracy.append_into(self.metrics)

		# print(self.metrics)
		# print(self.train_metrics)
		# print(self.val_metrics)
		# exit(0)

	def evaluate(self, model, features, targets, return_stats=False):

		prediction = model.forward(features)
		# print(f"prediction: {prediction.shape}:\n{prediction}")

		# Metrics actualisation
		for metric in self.eval_metrics:

			# print(f"val={self.val_on} -> {metric.name}")
			metric.reset(save=False)
			metric(prediction, targets)

			# if isinstance(metric, type(self.accuracy)):  #Opti with pre compute of val_accuracy
			# 	accuracy = metric.get_result()
			# 	print(f"accuracy find: {accuracy}")

		if return_stats:
			return self.loss.get_result(), self.accuracy.get_result()

	def fit(self,
			train_features,
			train_targets,
			batch_size,
			epochs,
			callbacks,
			val_features,
			val_targets,
			val_percent,
			verbose):

		if val_features and val_targets:
			print(f"Validation dataset is past")
			datasets = train_features, train_targets, val_features, val_targets
			self.val_on = True

		elif val_percent:
			# Split dataset in 2 -> New train dataset & validation dataset
			print(f"Split datasets in 2 batch with val_percent={val_percent}")
			datasets = Model.train_val_split(train_features, train_targets, val_percent)
			self.val_on = True

		else:
			print(f"No validation dataset: train dataset is using for validation")
			datasets = train_features, train_targets, train_features, train_targets
			self.val_on = False

		self.train_features = datasets[0]
		self.train_targets = datasets[1]
		self.val_features = datasets[2]
		self.val_targets = datasets[3]
		# self.train_features = datasets[0][:]
		# self.train_targets = datasets[1][:]
		# self.val_features = datasets[2][:]
		# self.val_targets = datasets[3][:]

		print(f"self.train_features: {self.train_features.shape}")
		print(f"self.train_targets: {self.train_targets.shape}")
		print(f"self.val_features: {self.val_features.shape}")
		print(f"self.val_targets: {self.val_targets.shape}")

		if self.val_on:
			self.current_metrics = list(self.metrics.values())
			self.eval_metrics = list(self.val_metrics.values())
		else:
			self.current_metrics = list(self.train_metrics.values())
			self.eval_metrics = list(self.train_metrics.values())

		print(f"Current metrics:\n{self.current_metrics}")
		print(f"Eval metrics:\n{self.eval_metrics}")
		print(f"Eval metrics:\n{self.eval_metrics}")
		# exit(0)

		# Train dataset length
		self.ds_train_len = len(self.train_features)

		# Batchs number
		self.n_batch = self.ds_train_len // batch_size + (1 if self.ds_train_len % batch_size else 0)

		# Split dataset into <n_batch> batch of len <batch_size>
		# self.batch_split = list(range(0, self.ds_train_len, batch_size))[1:]
		# print(self.batch_split)
		# exit(0)

		# Reset metrics for new model fit
		self.hard_reset_metrics()


	def get_metrics_logs(self):
		return ''.join(metric.log() for metric in self.current_metrics)

	def reset_metrics(self, save=False):
		for metric in self.current_metrics:
		# for metric in self.metrics.values():
			metric.reset(save)

	def hard_reset_metrics(self):
		for metric in self.current_metrics:
		# for metric in self.metrics.values():
			metric.hard_reset()

	def print_graph(self, metrics=[]):

		if metrics:
			metrics = [self.metrics[metric_name] for metric_name in metrics]
		else:
			metrics = self.current_metrics

		data = {}
		for metric in metrics:
			data[str(metric)] = metric.get_mem()

		data = {k:v for k, v in data.items() if len(v)}
		data['Subject goal'] = [0.08] * len(list(data.values())[0])
		data_df = pd.DataFrame(data)

		best_val = {key: max(values) if "accuracy" in key.lower() else min(values) for key, values in data.items()}

		print(data_df)
		print(f"Best metrics value: {best_val}")

		fig = px.line(data_df)
		fig.show()

	def summary(self, only_model_summary=True):

		print(f"\n-------------------")
		print(f"Summary of:\t{self.name}")

		self.optimizer.summary()
		for metric in self.train_metrics.values():
			metric.summary()

		if only_model_summary:
			print(f"-------------------\n")
