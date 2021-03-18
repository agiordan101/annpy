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

		# if self.sequence:
		# 	# Save reference of layers's weights matrix
		# 	self.weights.append(obj.init_weights(self.sequence[-1].output_shape))

		# Add Object into sequential model
		self.sequence.append(obj)

	def compile(self,
				loss="MSE",
				optimizer="SGD"):
		# Compile each layers
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
			self.weights.append(layer.compile(input_shape))
			input_shape = layer.output_shape


	def forward(self, inputs):

		for layer in self.sequence:
			inputs = layer.forward(inputs)

		return inputs

	# def fit(self,
	# 		features,
	# 		targets,
	# 		validation_features=None,
	# 		validation_targets=None,
	# 		k_fold_as_validation=False,
	# 		k_fold_percent=0.2,):
	# 	pass

	def summary(self):
		
		print(f"Summary of: {self.name}")
		print(f"Input shape:  {self.input_shape}")
		print(f"Output shape: {self.sequence[-1].output_shape}\n")

		for layer in self.sequence:
			layer.summary()

	def deepsummary(self):
		
		self.summary()
		print()
		for layer in self.weights:
			print(f"Weights:\n{layer}\n")

