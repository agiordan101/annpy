import annpy
from annpy.losses.MSE import MSE
from annpy.optimizers.SGD import SGD
from annpy.activations.ReLU import ReLU
from annpy.activations.Linear import Linear
from annpy.activations.Sigmoid import Sigmoid

objects = {
	'MSE': MSE,
	'SGD': SGD,
	'ReLU': ReLU,
	'Linear': Linear,
	'Sigmoid': Sigmoid,
}

def parse_object(obj, cls, str_allowed=True):

	# print(f"cls={cls}")
	if str_allowed and isinstance(obj, str) and obj in objects:
		obj = objects[obj]()

	if issubclass(type(obj), cls):
		return obj
	else:
		raise Exception(f"Object {obj} is not a child of abstact class {cls}")
