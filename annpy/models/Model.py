import annpy
from annpy.losses.Loss import Loss
from annpy.losses.BinaryCrossEntropy import BinaryCrossEntropy
from annpy.metrics.Metric import Metric
from annpy.metrics.Accuracy import Accuracy
from annpy.metrics.RangeAccuracy import RangeAccuracy
from annpy.optimizers.Optimizer import Optimizer

import numpy as np
import pandas as pd
import plotly.express as px

class Model():

	debug = []

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
		self.val_metrics_on = True # USELESS ??????

	def __str__(self):
		raise NotImplementedError

	def add_metric(self, metric):

		cpy = metric.copy().set_name('val_' + str(metric))
		self.metrics[str(cpy)] = cpy
		self.val_metrics[str(cpy)] = cpy

		self.metrics[str(metric)] = metric
		self.train_metrics[str(metric)] = metric

	def compile(self, loss, optimizer, metrics):

		if not isinstance(metrics, list):
			raise Exception("Error: Model: Metrics parameter in compile() is not a list")

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

	def forward(self):
		raise NotImplementedError

	def evaluate(self, model, features, target, val_metrics_on=True, return_stats=False):

		prediction = model.forward(features)

		# Metrics actualisation
		for metric in self.eval_metrics:

			# print(f"val={self.val_metrics_on} -> {metric.name}")
			metric.reset(save=False)
			metric(prediction, target)

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

		if self.val_on:
			self.current_metrics = list(self.metrics.values())
			self.eval_metrics = list(self.val_metrics.values())
		else:
			self.current_metrics = list(self.train_metrics.values())
			self.eval_metrics = list(self.train_metrics.values())

		# Batchs number
		self.last_batch_size = len(self.train_features) % batch_size
		self.n_batch_full = len(self.train_features) // batch_size
		self.n_batch = self.n_batch_full + 1 if self.last_batch_size else self.n_batch_full
		print(f"batch_size: {batch_size}\tn_batch_full: {self.n_batch_full}\tlast_batch_size: {self.last_batch_size}")

		# Reset metrics for new model fit
		self.hard_reset_metrics()

	def batchs_split(self, shuffle=True):

		# Shuffle
		if shuffle:
			a, b = Model.shuffle_datasets(self.train_features, self.train_targets, copy=False)

		if self.last_batch_size:
			last_f = a[-self.last_batch_size:]
			last_t = b[-self.last_batch_size:]

			a = a[:-self.last_batch_size]
			b = b[:-self.last_batch_size]

			# Split batches
			a = np.array_split(a, self.n_batch_full)
			b = np.array_split(b, self.n_batch_full)

			a.append(last_f)
			b.append(last_t)

		else:
			# Just Split batches
			a = np.array_split(a, self.n_batch_full)
			b = np.array_split(b, self.n_batch_full)

		# Merge
		return list(zip(a, b))

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
