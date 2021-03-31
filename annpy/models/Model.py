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

		self.metrics = {}
		self.train_metrics = {}
		self.val_metrics = {}
		self.loss = None
		self.optimizer = None

		self.stop_trainning = False
		self.val_metrics_on = True

	def __str__(self):
		raise NotImplementedError

	def add_metric(self, metric):

		cpy = metric.copy().set_name('val_' + str(metric))
		self.metrics[str(cpy)] = cpy
		self.val_metrics[str(cpy)] = cpy

		self.metrics[str(metric)] = metric
		self.train_metrics[str(metric)] = metric


	def compile(self, loss, optimizer, metrics):

		self.optimizer = annpy.utils.parse.parse_object(optimizer, Optimizer)

		# Add loss to metrics
		self.loss = annpy.utils.parse.parse_object(loss, Loss)
		self.add_metric(self.loss)
		# self.loss.append_into(self.metrics)

		# Save in accuracy the first metric to inherite from Accuracy
		self.accuracy = None

		for metric in metrics:
			metric = annpy.utils.parse.parse_object(metric, Metric)
			self.add_metric(metric)
			# metric.append_into(self.metrics)

			if not self.accuracy and issubclass(type(metric), Accuracy):
				self.accuracy = metric

		if not self.accuracy:
			self.accuracy = Accuracy()
			self.add_metric(self.accuracy)
			# self.accuracy.append_into(self.metrics)

		print(self.metrics)
		print(self.train_metrics)
		print(self.val_metrics)


	def forward(self):
		raise NotImplementedError
	
	def evaluate(self, model, features, target, val_metrics_on=True, verbose=True):

		prediction = model.forward(features)

		current_metrics = self.val_metrics.values() if self.val_metrics_on else self.train_metrics.values()

		# Metrics actualisation (Loss actualisation too)
		for metric in current_metrics:

			print(f"val={self.val_metrics_on} -> {metric.name}")
			metric.reset(save=False)
			metric(prediction, target)
			if isinstance(metric, type(self.loss)):
				loss = metric.get_result()
				print(f"loss find: {loss}")
			if isinstance(metric, type(self.accuracy)):
				accuracy = metric.get_result()
				print(f"accuracy find: {accuracy}")

		return loss, accuracy

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

		self.val_metrics_on = bool(val_percent)

		self.train_features = train_features
		self.train_targets = train_targets
		self.val_features = train_features
		self.val_targets = train_targets

		if val_features is None and val_percent is not None:
			# Split dataset in 2 -> New train dataset & validation dataset
			datasets = self.split_dataset(train_features, train_targets, [int(val_percent * len(train_features))])

			self.train_features = datasets[1][0]
			self.train_targets = datasets[1][1]
			self.val_features = datasets[0][0]
			self.val_targets = datasets[0][1]

		# Train dataset length
		self.ds_train_len = len(self.train_features)

		# Batchs number
		self.n_batch = self.ds_train_len // batch_size + (1 if self.ds_train_len % batch_size else 0)

		# Split dataset into <n_batch> batch of len <batch_size>
		self.batch_split = list(range(0, self.ds_train_len, batch_size))[1:]

		# Reset metrics for new model fit
		self.hard_reset_metrics()


	def get_metrics_logs(self):
		return ''.join([metric.log() for metric in self.metrics.values()])

	def reset_metrics(self, save=False):
		for metric in self.metrics.values():
			metric.reset(save)

	def hard_reset_metrics(self):
		for metric in self.metrics.values():
			metric.hard_reset()

	# def get_metrics(self):
	# 	for metric in self.metrics.values():
	# 		if self.val_metrics_on == ('val_' == metric.name[:4]):
	# 			yield metric

	def split_dataset(self, a, b, batch_split):

		# print(f"Split dataset: {batch_split}")
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

	def print_graph(self, metrics=[]):

		if metrics:
			metrics = [self.metrics[metric_name] for metric_name in metrics]
		else:
			metrics = list(self.metrics.values())

		data = {}
		for metric in metrics:
			data[str(metric)] = metric.get_mem()

		data_df = pd.DataFrame(data)
		print(data_df)

		fig = px.line(data_df)
		fig.show()


	def summary(self, only_model_summary=True):

		print(f"\n-------------------")
		print(f"Summary of:\t{self.name}")

		self.optimizer.summary()
		for metric in self.metrics.values():
			metric.summary()

		if only_model_summary:
			print(f"-------------------\n")
