from annpy.initializers.ConstantInitializer import ConstantInitializer

class Zeros(ConstantInitializer):

	def __init__(self):
		super().__init__(0.0)
