import numpy as np


class Model():

	def __init__(self,
					input_shape=None,
					input_layer=None,
					name="Default models name"):

		self.name = name
		self.input_shape = input_shape
		if input_layer:
			self.sequence = [input_layer]
		else:
			self.sequence = []

	def add(self, obj):
		# Add Object into sequential model
		self.sequence.append(obj)

	def compile(self,
				loss="MSE",
				optimizer="SGD"):
		# Save param for losses and optimizers

		if self.input_shape:
			pass

		elif self.sequence[0].input_shape:
			self.input_shape = self.sequence[0].input_shape

		else:
			raise Exception(f"[ERROR] {self} input_shape of layer 0 missing")

		input_shape = self.input_shape

		self.weights = []
		for layer in self.sequence:
			# print(f"New layer to compile {input_shape}")
			self.weights.extend(layer.compile(input_shape))
			input_shape = layer.output_shape

	def forward(self, inputs):

		# print(f"Input shape={inputs.shape}")
		# inputs = np.insert(inputs, len(inputs), 1, axis=len(inputs.shape) - 1)
		# print(inputs)
		for layer in self.sequence:
			inputs = layer.forward(inputs)

		return inputs

	def fit(self,
			features,
			targets,
			validation_features=None,
			validation_targets=None,
			k_fold_as_validation=False,
			k_fold_percent=0.2):

		prediction = self.forward(features)

		dloss = prediction - targets
		# self.sequence_rev = self.sequence.reverse() 


	def summary(self):
		
		print(f"Summary of: {self.name}")
		print(f"Input shape:  {self.input_shape}")
		print(f"Output shape: {self.sequence[-1].output_shape}\n")

		for layer in self.sequence:
			layer.summary()

	def deepsummary(self):
		
		self.summary()
		print()
		for i, layer in enumerate(self.weights):
			print(f"{'Bias' if i % 2 else 'Weights'} {i // 2}:\n{layer}\n")

