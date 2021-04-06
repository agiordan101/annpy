from annpy.optimizers.Optimizer import Optimizer

class Momentum(Optimizer):

	def __init__(self, lr=0.1):
		super().__init__(lr=lr)

	def __call__(self, deriv):
		# print(f"weights {weights.shape}:\n{weights}")
		# ret = -self.lr * deriv
		# ret = weights - self.lr * deriv
		# print(f"Shapes: {weights.shape} - {self.lr} * {deriv.shape} = {ret.shape}")
		# return ret
		return -self.lr * deriv

	# Method a passe dans lobject optimizer ? Clairement
	def apply_gradients(self, weights_lst):
		# weights_lst:	[[w0, b0], [..., ...], [wn, bn]]
		# gradients:	[(dx, dw, db), ...]

		for weightsb, gradients in zip(weights_lst[::-1], self.gradients):
			# print(f"weightsb {len(weightsb)}: {weightsb}")
			# print(f"")
			# print(f"Shapes w {weightsb[0].shape}:\n{weightsb[0]})")
			# print(f"Shapes b {weightsb[1].shape}:\n{weightsb[1]})")
			# print(f"Shapes dw {gradients[1].shape} / type {type(gradients[1][0][0])}:\n{gradients[1]}")
			# print(f"Shapes db {gradients[2].shape} / type {type(gradients[2][0])}:\n{gradients[2]}")
			# weightsb[0] = None
			# weightsb[1] = None

			weightsb[0] += self(gradients[1])
			weightsb[1] += self(gradients[2])

	def summary(self):
		print(f"Optimizer:\tannpy.optimizers.SGD, lr={self.lr}")

	# def __str__(self):
	# 	return "SGD"
