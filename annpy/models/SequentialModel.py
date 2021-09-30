import annpy
from annpy import activations, layers
from annpy.losses.Loss import Loss
from annpy.layers.Layer import Layer
from annpy.metrics.Metric import Metric
from annpy.metrics.Accuracy import Accuracy
from annpy.metrics.RangeAccuracy import RangeAccuracy
from annpy.optimizers.Optimizer import Optimizer

# from abc import classmethod
import json
import random
import numpy as np
import pandas as pd
import plotly.express as px
import copy

class SequentialModel():

	@classmethod
	def shuffle_datasets(cls, a, b, seed=None, copy=False):
		# shuffle_datasets needs to be used without SequentialModel
		"""
			If seed is None: shuffle_datasets doesn't care about NumPy seed (?)
			If not: shuffle_datasets need to not modify current NumPy seed
		"""
		if copy:
			a = a.copy()
			b = b.copy()

		if seed:
			# actual_seed = np.random.get_state()
			np.random.set_state(seed)
			shuffle_seed = seed
		else:
			shuffle_seed = np.random.get_state()

		np.random.shuffle(a)
		np.random.set_state(shuffle_seed)
		np.random.shuffle(b)

		return a, b

	@classmethod
	def train_test_split(cls, features, targets, val_percent, shuffle=True, tts_seed=None):
		# train_test_split needs to be used without SequentialModel

		# if tts_seed:
		# 	print(f"SequentialModel constructor: tts_seed: {tts_seed[1][:4]}")
		if shuffle:

			# Re seed numpy. Allow independence between tts_seed and w_seed if w_seed=True & tts_seed=False
			if not tts_seed:
				np.random.seed(random.choice(np.random.get_state()[1]) % 1000000)
				tts_seed = np.random.get_state()

			features, targets = cls.shuffle_datasets(features, targets, seed=tts_seed)

		i = int(val_percent * len(features))
		return features[i:, :], targets[i:, :], features[:i, :], targets[:i, :], tts_seed

	name: str
	input_shape: int

	weights:		list		# list of layers: [[w0, b0], [..., ...], [wn, bn]]
	sequence:		list		# list of Object: [L0, ..., Ln]
	sequence_rev:	list		# list of Object: [L0, ..., Ln]

	train_features:		np.ndarray
	train_targets:		np.ndarray
	val_features:		np.ndarray
	val_targets:		np.ndarray
	last_batch_size:	int		# last full batch size index

	loss:		Loss = None
	optimizer:	Optimizer = None
	accuracy:	Metric = None

	metrics:		dict		# All metrics used for all fitting (train & validation)

	val_on:			bool = True
	stop_trainning:	bool = False
	fit_ret:		dict = None

	def __init__(self,
					input_shape=None,
					input_layer=None,
					name="default_model_name",
					seed=None,
					tts_seed=None):

		self.name = name
		self.input_shape = input_shape
		self.weights = []

		self.w_seed = seed
		self.tts_seed = tts_seed
		# if tts_seed:
		# 	print(f"SequentialModel constructor: tts_seed: {tts_seed[1][:4]}")

		if input_layer:
			self.sequence = [input_layer]
		else:
			self.sequence = []

		self.metrics = {}
		# self.loss = None
		# self.optimizer = None
		# self.accuracy = None

		# self.stop_trainning = False
		# self.val_on = True

	def __str__(self):
		return "SequentialModel"

	def add(self, layer):

		if issubclass(type(layer), Layer):
			# Add Object into sequential model
			layer.set_layer_index(len(self.sequence))
			self.sequence.append(layer)

		else:
			raise Exception(f"[annpy error]:Object {layer} is not a child of abstact class {Layer}")

	def add_metric(self, metric):

		cpy = metric.copy().set_name('val_' + str(metric))
		self.metrics[str(cpy)] = cpy
		self.metrics[str(metric)] = metric

	def compile_metrics(self, loss, metrics):

		# Add loss as metric
		if isinstance(loss, str):
			self.add_metric(annpy.parsing.parse.parse_object(loss, Loss))
		else:
			self.add_metric(loss)

		# Parse metrics
		for metric in metrics:
			self.add_metric(annpy.parsing.parse.parse_object(metric, Metric))

	def compile(self,
				loss="MSE",
				optimizer="Adam",
				optimizer_kwargs={},
				metrics=[]):

		# -- MODEL --

		if not isinstance(metrics, list):
			raise TypeError("[annpy error] Metrics parameter in SequentialModel.compile() is not a List instance")
		if not isinstance(optimizer_kwargs, dict):
			raise TypeError("[annpy error] optimizer_kwargs parameter in SequentialModel.compile() is not a Dict instance")

		# Parse optimizer, loss
		self.optimizer = annpy.parsing.parse.parse_object(optimizer, Optimizer, **optimizer_kwargs)
		self.loss = annpy.parsing.parse.parse_object(loss, Loss)
		self.compile_metrics(self.loss, metrics)

		# -- SEQUENTIAL --

		# Handling input_shape
		if self.input_shape:
			pass

		elif len(self.sequence) and self.sequence[0].input_shape:
			self.input_shape = self.sequence[0].input_shape

		else:
			raise Exception(f"[annpy error] SequentialModel.compile(): input_shape of layer 0 missing")

		# Set NumPy seed for kernel/bias initialisation
		if self.w_seed:
			# print(f"Set NumPy seed: {self.w_seed[1][:4]}")
			np.random.set_state(self.w_seed)
		else:
			self.w_seed = np.random.get_state()
			# print(f"Get NumPy seed: {self.w_seed[1][:4]}")

		input_shape = self.input_shape
		for layer in self.sequence:

			# Compile all layers
			weights = layer.compile(input_shape)
			self.weights.append(weights)

			# Save weights references in optimizer
			self.optimizer.add(weights)

			# Save next input shape
			input_shape = layer.output_shape

		# Compile optimizer
		self.optimizer.compile()

		# Make reverse list for fitting method
		self.sequence_rev = self.sequence.copy()
		self.sequence_rev.reverse()

	def get_seeds(self):
		return {
			'w_seed': self.w_seed,
			'tts_seed': self.tts_seed,
		}

	def forward(self, inputs):
		try:
			for layer in self.sequence:
				inputs = layer.forward(inputs)
			return inputs
		except Exception as error:
			print(f"[annpy error] SequentialModel.forward(): Error in this model structure ?\n{error}")
			exit(0)

	def evaluate(self, features, target, metrics_on=True, return_stats=False, only_val=True):
		# def evaluate(self, model, features, target, metrics_on=True, return_stats=False):

		prediction = self.forward(features)
		# prediction = model.forward(features)
		# print(f"metrics 0: {self.metrics}")

		# Metrics actualisation
		for metric in self.metrics.values():
			if not only_val or 'val_' in str(metric):
				metric.reset(save=False)
				metric(prediction, target)
				# print(f"{str(metric)}: count / total {metric.count} / {metric.total}")
				# print(f"metric: {metric} / {metric.get_mem()}")

		# print(f"metrics 1: {self.metrics}")
		if return_stats:
			return {key: metric.get_result() for key, metric in self.metrics.items() if not only_val or 'val_' in key}


	def dataset_fit_setup(self, train_features, train_targets, val_features, val_targets, val_percent):

		if val_features and val_targets:
			# print(f"Validation dataset is past")
			datasets = train_features, train_targets, val_features, val_targets

		# Split train dataset in two parts
		elif val_percent:
			# print(f"Split datasets in 2 batch with val_percent={val_percent}")
			datasets = SequentialModel.train_test_split(train_features, train_targets, val_percent, tts_seed=self.tts_seed)
			if not self.tts_seed:
				self.tts_seed = datasets[4]

		else:
			# print(f"No validation dataset: train dataset is using for validation")
			datasets = train_features, train_targets, train_features, train_targets
			self.val_on = False

		self.train_features = datasets[0]
		self.train_targets = datasets[1]
		self.val_features = datasets[2]
		self.val_targets = datasets[3]

	def batchs_split(self, shuffle=True):

		a, b = self.train_features, self.train_targets

		# Shuffle
		if shuffle:
			a, b = SequentialModel.shuffle_datasets(self.train_features, self.train_targets)

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

	def fit(self,
			train_features,
			train_targets,
			batch_size=42,
			epochs=420,
			callbacks=[],
			val_features=None,
			val_targets=None,
			val_percent=0.2,
			tts_seed=None,
			verbose=True,
			print_graph=False):

		train_features = copy.deepcopy(train_features)
		train_targets = copy.deepcopy(train_targets)

		# Parse datasets
		if tts_seed:
			self.tts_seed = tts_seed
			print(f"tts_seed: {tts_seed[1][:4]}")
			
		self.dataset_fit_setup(train_features, train_targets, val_features, val_targets, val_percent)

		# Batchs stats
		self.last_batch_size = len(self.train_features) % batch_size
		self.n_batch_full = len(self.train_features) // batch_size
		self.n_batch = self.n_batch_full + 1 if self.last_batch_size else self.n_batch_full
		# print(f"batch_size: {batch_size}\tn_batch_full: {self.n_batch_full}\tlast_batch_size: {self.last_batch_size}")

		# Reset metrics for new model fit
		self.hard_reset_metrics()

		# Callbacks TRAIN begin
		for cb in callbacks:
			cb.on_train_begin(model=self)

		for epoch in range(epochs):

			if verbose:
				print(f"EPOCH {epoch}")

			# Callbacks EPOCH begin
			for cb in callbacks:
				cb.on_epoch_begin()

			# Dataset shuffle + split
			batchs = self.batchs_split()

			# for step, (features, target) in enumerate(batchs):
			for step, data in enumerate(batchs):

				# if verbose:
				# 	print(f"STEP={step}/{self.n_batch - 1}")
				# 	print(f"STEP={step}/{self.n_batch - 1}\tloss: {self.loss.get_result()}")

				# Callbacks BATCH begin
				for cb in callbacks:
					cb.on_batch_begin()

				features, target = data

				# Prediction
				prediction = self.forward(features)

				# Metrics actualisation
				for metric in self.metrics.values():
					if 'val_' not in str(metric):
						metric(prediction, target)

				# Backpropagation
				dx = self.loss.derivate(prediction, target)

				# For sequential model: dw, db
				gradients = [0, 0]
				
				self.optimizer.gradients = []
				for layer in self.sequence_rev:
					# print(f"LAYER {layer.layer_index}")
					dx, gradients = layer.backward(dx)
					# gradients = layer.backward(gradients[0])
					self.optimizer.gradients.append(gradients)

				# Optimizer
				self.optimizer.apply_gradients(self.weights)

				# Callbacks BATCH end
				for cb in callbacks:
					cb.on_batch_end()

			self.evaluate(self.val_features, self.val_targets)
			# self.evaluate(self, self.val_features, self.val_targets)

			if verbose:
				# Get total metrics data of this epoch
				print(f"Metrics: {self.get_metrics_logs()}")
				print(f"\n-------------------------\n")

			# Callbacks EPOCH end
			for cb in callbacks:
				cb.on_epoch_end(verbose=verbose)

			# Save in mem & Reset metrics values
			self.reset_metrics()
			# print(f"metric: {list(self.metrics.values())[0].get_result()}")
			# print(f"metric: {list(self.metrics.values())[0].get_mem()}")

			if self.stop_trainning:
				break

		if print_graph:
			self.print_graph()

		# Callbacks TRAIN end
		for cb in callbacks:
			cb.on_train_end()

		self.fit_ret = {key: metric.get_mem() for key, metric in self.metrics.items()}
		return self.fit_ret


	def get_metrics_logs(self):
		return ''.join(metric.log() for metric in self.metrics.values())

	def reset_metrics(self, save=True):
		for metric in self.metrics.values():
			metric.reset(save)

	def hard_reset_metrics(self):
		# print(f"Metrics:\n{self.metrics}")
		for metric in self.metrics.values():
			metric.hard_reset()

	def print_graph(self, metrics=[]):
		# Re work this horrible function

		if metrics:
			metrics = [self.metrics[metric_name] for metric_name in metrics]
		else:
			metrics = self.metrics.values()

		data = {str(metric) : metric.get_mem() for metric in metrics if metric.get_mem()}
		data['Subject goal'] = [0.08] * len(list(data.values())[0])
		data_df = pd.DataFrame(data)

		print(data_df)

		# best_val = {key: max(values) if "accuracy" in key.lower() else min(values) for key, values in data.items()}
		# print(f"Best metrics value: {best_val}")

		fig = px.line(data_df)
		fig.show()

	"""
		Summaries
	"""

	def summary(self, only_model_summary=True):

		print(f"Input shape:\t{self.input_shape}")
		print(f"Output shape:\t{self.sequence[-1].output_shape}\n")
		for layer in self.sequence:
			layer.summary()

		if only_model_summary:
			print(f"-------------------\n")

	def deepsummary(self, model_summary=True):
		
		if model_summary:
			self.summary(only_model_summary=False)

		for i, layer in enumerate(self.weights):
			print(f"\nLayer {i}:\n")
			print(f"Weights {layer[0].shape}:\n{layer[0]}\n")
			print(f"Bias {layer[1].shape}:\n{layer[1]}\n")

		print(f"-------------------\n")

	"""
		Model save
	"""

	def _save(self):
		return {
			'type': "SequentialModel",
			'name': self.name,
			'input_shape': self.input_shape,
			'layers': [layer._save() for layer in self.sequence]
		}

	def save_weights(self, folder_path, file_name=None):
		"""
			Specific configuration to only save weights
		"""

		file_name = file_name or f"{self.name}_weights"
		struct = {
			'file_type': "Only weights",
			'model': self._save()
		}

		file_path = f"{folder_path}/{file_name}.json"
		with open(file_path, 'w') as f:
			# f.write(struct)
			json.dump(struct, f, indent=4)
			print(f"Successfully save model at {file_path}")

		return file_path, struct

	@classmethod
	def load_model(obj, file_path):

		model = None

		try:
			with open(file_path, 'r') as f:
				data = json.loads(f.read())

				if data['file_type'] != "Only weights":
					raise Exception(f"[annpy error] load_model: Wrong <file_type> for file {file_path}")

				data = data['model']

				if data['type'] != obj.__name__:
					raise Exception(f"[annpy error] SequentialModel.load_model(): Wrong model type in {file_path} for this classmethod")

				model = SequentialModel(
					input_shape=data['input_shape'],
					name=data['name']
				)

				for layer in data['layers']:

					model.add(annpy.parsing.parse.parse_object(
						layer['type'],
						Layer,
						output_shape=layer['units'],
						activation=layer['activation'],
						kernel=layer['kernel'],
						bias=layer['bias'],
						name=layer['name']
					))
				
				print(f"Successfully load model at '{file_path}'")

		except FileNotFoundError:
			print(f"[annpy error] SequentialModel.load_model(): File not found at '{file_path}'")
			exit(1)

		except Exception as error:
			print(f"[annpy error] SequentialModel.load_model(): Error in model file structure ({file_path}):\n{error}")
			exit(1)

		return model
