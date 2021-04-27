import annpy
import numpy as np

from annpy.models.Model import Model
from annpy.layers.Layer import Layer
# from annpy.metrics.Metric import metrics_data_to_str

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

	def __str__(self):
		return "Sequential"

	def add(self, obj):
		if issubclass(type(obj), Layer):
			# Add Object into sequential model
			obj.set_layer_index(len(self.sequence))
			# obj.layer_index = len(self.sequence)
			self.sequence.append(obj)
		else:
			raise Exception(f"Object {obj} is not a child of abstact class {Layer}")

	def compile(self,
				loss="MSE",
				optimizer="Adam",
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

		# self.weightsB:	[[w0, b0], [..., ...], [wn, bn]]
		self.weightsB = []
		for layer in self.sequence:
			# print(f"New layer to compile {input_shape}")
			weightsB = layer.compile(input_shape)
			self.weightsB.append(weightsB)

			# Save next input shape
			input_shape = layer.output_shape

			# Add matrix for optimizer math
			self.optimizer.add(weightsB)

		# Optimizers maths
		self.optimizer.compile()

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
			val_features=None,
			val_targets=None,
			val_percent=0.2,
			verbose=True,
			print_graph=True):

		super().fit(train_features, train_targets, batch_size, epochs, callbacks, val_features, val_targets, val_percent, verbose)

		# Callbacks TRAIN begin
		for cb in callbacks:
			cb.on_train_begin()

		for epoch in range(epochs):

			print(f"EPOCH {epoch}")

			# Callbacks EPOCH begin
			for cb in callbacks:
				cb.on_epoch_begin()

			# print(f"self.train_features: {self.train_features.shape}")
			# print(f"self.train_targets: {self.train_targets.shape}")
			# Dataset shuffle + split
			batchs = Model.split_dataset_batches(self.train_features, self.train_targets, self.n_batch)
			# batchs = Model.split_dataset_batches(train_features, train_targets, self.batch_split, self.n_batch)

			# print(list(batchs))
			# for step, (features, target) in enumerate(batchs):
			for step, data in enumerate(batchs):

				if verbose:
					print(f"STEP={step}/{self.n_batch - 1}")
					# print(f"STEP={step}/{self.n_batch - 1}\tloss: {self.loss.get_result()}")

				# Callbacks BATCH begin
				for cb in callbacks:
					cb.on_batch_begin()

				features, targets = data
				print(f"features: {features.shape}")
				print(f"targets: {targets.shape}")

				# Prediction
				prediction = self.forward(features)

				# Metrics actualisation
				for metric in self.train_metrics.values():
					# print(f"train={self.val_metrics_on} -> {metric.name}")
					metric(prediction, targets)

				# Backpropagation
				dx = self.loss.derivate(prediction, targets)
				gradients = [0, 0]
				self.optimizer.gradients = []
				for layer in self.sequence_rev:
					# print(f"LAYER {layer.layer_index}")
					dx, gradients = layer.backward(dx)
					# gradients = layer.backward(gradients[0])
					self.optimizer.gradients.append(gradients)

				# Optimizer
				self.optimizer.apply_gradients(self.weightsB)
				# exit(0)

				# Callbacks BATCH end
				for cb in callbacks:
					cb.on_batch_end()

			# self.debug.append(list(self.train_metrics.values())[0].get_result())
			
			self.evaluate(self, self.val_features, self.val_targets)
			# print(f"self.val_features: {self.val_features.shape}")
			# print(f"self.val_targets: {self.val_targets.shape}")
			# exit()

			# Get total metrics data of this epoch
			print(f"Metrics: {self.get_metrics_logs()}")
			print(f"\n-------------------------\n")

			# Callbacks EPOCH end
			for cb in callbacks:
				cb.on_epoch_end(verbose=verbose)

			# Save in mem & Reset metrics values
			self.reset_metrics(save=True)

			if self.stop_trainning:
				break

		if print_graph:
			self.print_graph()

		# Callbacks TRAIN end
		for cb in callbacks:
			cb.on_train_end()

		return self.loss.get_mem()[-1], self.accuracy.get_mem()[-1]


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

		for i, layer in enumerate(self.weightsB):
			print(f"\nLayer {i}:\n")
			print(f"Weights {layer[0].shape}:\n{layer[0]}\n")
			print(f"Bias {layer[1].shape}:\n{layer[1]}\n")

		print(f"-------------------\n")
