class SGD():

	def __init__(self, lr=0.1):
		self.lr = lr
	
	def __call__(self, weights, deriv):
		return weights - self.lr * deriv
