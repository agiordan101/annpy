import annpy
import numpy as np
import pandas as pd
import plotly.express as px

from sklearn.model_selection import train_test_split

from annpy.losses.Loss import Loss
from annpy.metrics.Metric import Metric
from annpy.metrics.Accuracy import Accuracy
from annpy.optimizers.Optimizer import Optimizer


from annpy.losses.BinaryCrossEntropy import BinaryCrossEntropy
from annpy.metrics.RangeAccuracy import RangeAccuracy


class Model():

	debug = []

	def __init__(self, input_shape, input_layer, name):

		self.name = name
		self.input_shape = input_shape
		self.weights = None

		self.metrics = {}				# All metrics used for all fitting (train & validation)
		self.train_metrics = {}			# Metrics use on train dataset
		self.val_metrics = {}			# Metrics use on valid dataset
		self.current_metrics = {}		# Metrics use on train/valid dataset.s in one fitting (If no validation dataset is asked, same references as self.train_metrics, overwise as self.metrics)
		self.eval_metrics = {}			# Metrics use to validate each epoch in one fitting

		self.loss = None
		self.optimizer = None
		self.accuracy = None

		self.stop_trainning = False
		self.val_metrics_on = True


		# self.val_loss = BinaryCrossEntropy()
		# self.val_accuracy = RangeAccuracy([0.5, 0.5])


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

	def forward(self):
		raise NotImplementedError

	# def evaluate(self, model, features, target, val_metrics_on=True, return_stats=False):

	# 	prediction = model.forward(features)

	# 	self.val_loss(prediction, target)
	# 	self.val_accuracy(prediction, target)

	# 	self.val_loss.reset(save=True)
	# 	self.val_accuracy.reset(save=True)



	def evaluate(self, model, features, target, val_metrics_on=True, return_stats=False):

		prediction = model.forward(features)
		# accuracy = None

		# Loss actualisation
		# loss_metric = self.eval_metrics[0]
		# loss_metric.reset(save=False)
		# loss_metric(prediction, target)
		# loss = loss_metric.get_result()
		# print(f"loss find: {type(loss_metric)} -> {loss}")

		# print(f"val ds len: {len(features)}")

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

		self.val_metrics_on = bool(val_percent) # merge 2 lines ?
		# self.current_metrics = list(self.metrics.values() if self.val_metrics_on else self.train_metrics.values())
		# self.eval_metrics = list(self.val_metrics.values() if self.val_metrics_on else self.train_metrics.values())

		if self.val_metrics_on:
			self.current_metrics = list(self.metrics.values())
			self.eval_metrics = list(self.val_metrics.values())
		else:
			self.current_metrics = list(self.train_metrics.values())
			self.eval_metrics = list(self.train_metrics.values())

		# print(self.current_metrics)
		# print(self.train_metrics)
		# print(self.val_metrics)
		# exit(0)

		self.train_features = train_features
		self.train_targets = train_targets
		self.val_features = train_features
		self.val_targets = train_targets

		if val_features is None and val_percent is not None:
			# Split dataset in 2 -> New train dataset & validation dataset
			self.n_batch = 2
			datasets = self.split_dataset(train_features, train_targets, [int(val_percent * len(train_features))])
			self.train_features = datasets[1][0]
			self.train_targets = datasets[1][1]
			self.val_features = datasets[0][0]
			self.val_targets = datasets[0][1]

			# X_train, X_test, y_train, y_test = train_test_split(train_features, train_targets, test_size=0.2, shuffle=False)
			# self.train_features = X_train
			# self.train_targets = y_train
			# self.val_features = X_test
			# self.val_targets = y_test

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
		return ''.join([metric.log() for metric in self.current_metrics])

	def reset_metrics(self, save=False):
		for metric in self.current_metrics:
		# for metric in self.metrics.values():
			metric.reset(save)

	def hard_reset_metrics(self):
		for metric in self.current_metrics:
		# for metric in self.metrics.values():
			metric.hard_reset()


	def split_dataset(self, a, b, shuffle=True):

		# Shuffle
		if shuffle:
			a = a.copy()
			b = b.copy()
			seed = np.random.get_state()
			np.random.shuffle(a)
			np.random.set_state(seed)
			np.random.shuffle(b)
		
		# Split batches
		a = np.array_split(a, self.n_batch)
		b = np.array_split(b, self.n_batch)

		# Merge
		return list(zip(a, b))



	# def print_graph(self, metrics=[]):
		
	# 	data = {
	# 		'loss': self.val_loss.get_mem(),
	# 		'accuracy': self.val_accuracy.get_mem()
	# 	}
	# 	data_df = pd.DataFrame(data)

	# 	fig = px.line(data_df)
	# 	fig.show()

	def print_graph(self, metrics=[]):

		if metrics:
			metrics = [self.metrics[metric_name] for metric_name in metrics]
		else:
			metrics = self.current_metrics

		data = {}
		for metric in metrics:
			data[str(metric)] = metric.get_mem()

		# print(len(self.debug))
		data = {k:v for k, v in data.items() if len(v)}
		data['Subject goal'] = [0.08] * len(list(data.values())[0])
		# data['debug'] = self.debug
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
		for metric in self.metrics.values():
			metric.summary()

		if only_model_summary:
			print(f"-------------------\n")
