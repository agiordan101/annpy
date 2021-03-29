import annpy
import numpy as np

from annpy.models.Model import Model
from annpy.layers.Layer import Layer
from annpy.metrics.Metric import metrics_data_to_str

class Sequencial(Model):

	def __init__(self,
					input_shape=None,
					input_layer=None,
					name="Default models name"):

		super().__init__(input_shape, input_layer, name)

		if input_layer:
			self.sequence = [input_layer]
		else:
			self.sequence = []

	# def __str__(self):
	# 	return "Sequential"

	def add(self, obj):
		if issubclass(type(obj), Layer):
			# Add Object into sequential model
			obj.layer_index = len(self.sequence)
			self.sequence.append(obj)
		else:
			raise Exception(f"Object {obj} is not a child of abstact class {Layer}")

	def compile(self,
				loss="MSE",
				optimizer="SGD",
				metrics=[]):
		
		# Save loss, optimizer and metrics
		super().compile(loss, optimizer, metrics)

		# input_shape handler
		if self.input_shape:
			pass
		elif self.sequence[0].input_shape:
			self.input_shape = self.sequence[0].input_shape
		else:
			raise Exception(f"[ERROR] {self} input_shape of layer 0 missing")
		input_shape = self.input_shape

		# w0, b0, ..., ..., wn, bn
		self.weights = []
		for layer in self.sequence:
			# print(f"New layer to compile {input_shape}")
			self.weights.append(layer.compile(input_shape))
			input_shape = layer.output_shape

		# Make reverse list for fitting method
		self.sequence_rev = self.sequence.copy()
		self.sequence_rev.reverse()


	def forward(self, inputs):

		# print(f"Input shape={inputs.shape}")
		# inputs = np.insert(inputs, len(inputs), 1, axis=len(inputs.shape) - 1)
		# print(inputs)
		for layer in self.sequence:
			inputs = layer.forward(inputs)

		return inputs

	def fit(self,
			train_features,
			train_targets,
			batch_size=42,
			epochs=420,
			callbacks=[],
			valid_features=None,
			valid_targets=None,
			valid_percent=0.2,
			verbose=True):

		super().fit(train_features, train_targets, batch_size, epochs, callbacks, valid_features, valid_targets, valid_percent, verbose)

		for metric in self.metrics:
			metric.hard_reset()

		for epoch in range(epochs):

			# Dataset shuffle + split
			batchs = super().split_dataset(train_features, train_targets, self.batch_split)

			# print(list(batchs))
			for step, data in enumerate(batchs):

				features, target = data

				# Prediction
				prediction = self.forward(features)

				# Metrics actualisation (Loss actualisation too)
				for metric in self.metrics:
					metric(prediction, target)

				if verbose:
					print(f"STEP={step}/{self.n_batch - 1}")
					# print(f"STEP={step}/{self.n_batch - 1}\tloss: {self.loss.get_result()}")

				# Backpropagation
				gradients = self.loss.derivate(prediction, target), 0, 0
				self.optimizer.gradients = []
				for layer in self.sequence_rev:
					# print(f"LAYER {layer.layer_index}")
					gradients = layer.backward(gradients[0])
					self.optimizer.gradients.append(gradients)

				# Optimizer
				self.optimizer.apply_gradients(self.weights)

			# Save loss/accuracy of this epoch
			# self.loss.save_result()
			# self.accuracy.save_result()

			# Get total metrics data of this epoch & reset vars
			metrics_log = metrics_data_to_str(self.metrics)

			print(f"\n-------------------------")
			print(f"EPOCH {epoch}/{epochs - 1}{metrics_log}")

		self.print_graph()

		return super().evaluate(self, self.valid_features, self.valid_targets, verbose=verbose)



	def summary(self, only_model_summary=True):
		
		super().summary(only_model_summary=only_model_summary)

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
