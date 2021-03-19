from annpy.activations.Activation import Activation

class Linear(Activation):

	def __init__(self):
		pass

	def __call__(self, x):
		return x

	def derivate(self, x):
		return x
