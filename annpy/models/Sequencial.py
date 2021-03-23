import annpy
import numpy as np
from annpy.models.Model import Model
from annpy.layers.Layer import Layer

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
				optimizer="SGD"):
		
		super().compile(loss, optimizer)

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

		# self.deepsummary()
		# exit(0)

	def forward(self, inputs):

		# print(f"Input shape={inputs.shape}")
		# inputs = np.insert(inputs, len(inputs), 1, axis=len(inputs.shape) - 1)
		# print(inputs)
		for layer in self.sequence:
			inputs = layer.forward(inputs)

		return inputs

	def split_dataset(self, a, b, array_split):

		# Shuffle
		seed = np.random.get_state()
		np.random.shuffle(a)
		np.random.set_state(seed)
		np.random.shuffle(b)

		# Split batches
		a = np.split(a, array_split)
		b = np.split(b, array_split)

		# Merge
		return list(zip(a, b))

	def fit(self,
			train_features,
			train_targets,
			batch_size=4,
			epochs=420,
			# validation_features=None,
			# validation_targets=None,
			# k_fold_as_validation=False,
			# k_fold_percent=0.2,
			verbose=True):

		# Dataset length
		features_len = len(train_features)

		# Batchs number
		n_batch = features_len // batch_size + (1 if len(train_features) % batch_size else 0)

		# Split index list
		array_split = list(range(0, features_len, batch_size))[1:]

		# print(f"vars {features_len} / {n_batch} / {array_split}")

		loss_sum = None
		for epoch in range(epochs):

			if verbose:
				print(f"\n-------------------------")
				print(f"EPOCH={epoch}/{epochs - 1}")

			# Dataset shuffle + split
			batchs = self.split_dataset(train_features, train_targets, array_split)

			# print(list(batchs))
			loss_sum = 0
			for step, data in enumerate(batchs):

				features, targets = data
				# print(f"data {data}")
				if verbose:
					print(f"-----------")
					print(f"STEP={step}/{n_batch - 1}")
					# print(f"features {features}")
					# print(f"targets {targets}")

				prediction = self.forward(features)

				loss = self.loss(prediction, targets)
				loss_sum += loss
				gradients = prediction - targets, 0, 0

				# print(f"loss function: {gradients[0]}")

				self.optimizer.gradients = []
				for layer in self.sequence_rev:
					# print(f"LAYER {layer.layer_index}")
					gradients = layer.backward(gradients[0])
					self.optimizer.gradients.append(gradients)
				

				if verbose:
					print(f"STEP {step} loss={loss}\n")

				# info = np.array(self.weights, dtype=object)
				# print(f"\nself.weights {info.shape}:\n{info}\n")

				# self.deepsummary()
				self.optimizer.update_weights(self.weights)
				# self.deepsummary()
				# exit(0)

				# info = np.array(self.weights, dtype=object)
				# print(f"self.weights {info.shape}:\n{info}\n\n")

			loss_sum /= n_batch
			print(f"EPOCH {epoch} loss={loss_sum}")
		
		return loss_sum



	def summary(self, only_model_summary=True):
		
		super().summary(only_model_summary=only_model_summary)

		print(f"Output shape: {self.sequence[-1].output_shape}\n")
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
