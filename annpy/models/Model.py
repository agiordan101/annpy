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
		self.loss = None
		self.optimizer = None

		self.stop_trainning = False

	def __str__(self):
		raise NotImplementedError

	def compile(self, loss, optimizer, metrics):

		self.optimizer = annpy.utils.parse.parse_object(optimizer, Optimizer)

		# Add loss to metrics
		self.loss = annpy.utils.parse.parse_object(loss, Loss)
		self.loss.append_into(self.metrics)

		# Save in accuracy the first metric to inherite from Accuracy
		self.accuracy = None

		for metric in metrics:
			metric = annpy.utils.parse.parse_object(metric, Metric)
			metric.append_into(self.metrics)

			if not self.accuracy and issubclass(type(metric), Accuracy):
				self.accuracy = metric

		if not self.accuracy:
			self.accuracy = Accuracy()
			self.accuracy.append_into(self.metrics)

		print(self.metrics)


	def forward(self):
		raise NotImplementedError
	
	def evaluate(self, model, features, target, verbose=True):

		prediction = model.forward(features)

		# self.loss.reset(save=False)
		# self.loss(predictions, targets)
		# loss = self.loss.get_result()

		# self.accuracy.reset(save=False)
		# self.accuracy(predictions, targets)
		# accuracy = self.accuracy.get_result()

		# Metrics actualisation (Loss actualisation too)
		for metric in self.metrics.values():
			if 'val_' in metric.name:
				metric.reset(save=False)
				metric(prediction, target)

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
		
		self.train_features = train_features
		self.train_targets = train_targets
		self.val_features = val_features
		self.val_targets = val_targets

		if val_features is None and val_percent is not None:
			# Split dataset in 2 -> New train dataset & validation dataset
			datasets = self.split_dataset(train_features, train_targets, [int(val_percent * len(train_features))])

			self.val_features = datasets[0][0]
			self.val_targets = datasets[0][1]
			self.train_features = datasets[1][0]
			self.train_targets = datasets[1][1]

		# Train dataset length
		self.ds_train_len = len(self.train_features)

		# Batchs number
		self.n_batch = self.ds_train_len // batch_size + (1 if self.ds_train_len % batch_size else 0)

		# Split dataset into <n_batch> batch of len <batch_size>
		self.batch_split = list(range(0, self.ds_train_len, batch_size))[1:]

		for metric in self.metrics.values():
			metric.hard_reset()


	def get_metrics_logs(self):
		return ''.join([metric.log() for metric in self.metrics.values()])

	def reset_metrics(self, save):
		for metric in self.metrics.values():
			metric.reset(save)

	def split_dataset(self, a, b, batch_split):

		print(f"Split dataset")
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
