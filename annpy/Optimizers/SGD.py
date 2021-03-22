from annpy.optimizers.Optimizer import Optimizer

class SGD(Optimizer):

	def __init__(self, lr=0.1):
		super().__init__(lr=lr)

	def __call__(self, weights, deriv):
		return weights - self.lr * deriv

	# def __str__(self):
	# 	return "SGD"