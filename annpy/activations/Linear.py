from annpy.activations.Activation import Activation

class Linear(Activation):

	def __call__(self, x):
		return x

	def derivate(self, x):
		return 1.

	def __str__(self):
		return "Linear"
