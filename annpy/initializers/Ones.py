from annpy.initializers.ConstantInitializer import ConstantInitializer

class Ones(ConstantInitializer):

	def __init__(self):
		super().__init__(1.0)
