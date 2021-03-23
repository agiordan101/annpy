from annpy.optimizers.Optimizer import Optimizer

class SGD(Optimizer):

	def __init__(self, lr=0.1):
		super().__init__(lr=lr)

	def __call__(self, weights, deriv):
		# print(f"weights {weights.shape}:\n{weights}")
		ret = weights - self.lr * deriv
		# print(f"Shapes: {weights.shape} - {self.lr} * {deriv.shape} = {ret.shape}")
		return ret

	def update_weights(self, weights_lst):
		# weights_lst:	[[w0, b0], [..., ...], [wn, bn]]
		# gradients:	[(dx, dw, db), ...]

		self.gradients.reverse()
		for weightsb, gradients in zip(weights_lst, self.gradients):
			# print(f"weightsb {len(weightsb)}: {weightsb}")
			print(f"Shapes: ({gradients[0].shape}, {gradients[1].shape}, {gradients[2].shape})")
			# weightsb[0] = None
			# weightsb[1] = None
			weightsb[0] = self(weightsb[0], gradients[1])
			weightsb[1] = self(weightsb[1], gradients[2])

	def summary(self):
		print(f"Optimizer:\t{self} ,\tlr={self.lr}")

	# def __str__(self):
	# 	return "SGD"
