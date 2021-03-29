import annpy
from annpy.losses.MSE import MSE
from annpy.optimizers.SGD import SGD
from annpy.metrics.Accuracy import Accuracy
from annpy.activations.ReLU import ReLU
from annpy.activations.Linear import Linear
from annpy.activations.Sigmoid import Sigmoid
from annpy.initializers.LecunNormal import LecunNormal
from annpy.initializers.LecunUniform import LecunUniform
from annpy.initializers.RandomNormal import RandomNormal
from annpy.initializers.RandomUniform import RandomUniform

objects = {
	'mse': MSE,
	'sgd': SGD,
	'relu': ReLU,
	'linear': Linear,
	'sigmoid': Sigmoid,
	'accuracy': Accuracy,
	'lecunnormal': LecunNormal,
	'lecununiform': LecunUniform,
	'randomnormal': RandomNormal,
	'randomuniform': RandomUniform,
}

def parse_object(obj, cls, str_allowed=True):

	# print(f"cls={cls}")
	if str_allowed and isinstance(obj, str) and obj.lower() in objects:
		obj = objects[obj.lower()]()

	if issubclass(type(obj), cls):
		return obj
	else:
		raise Exception(f"Object {obj} is not a child of abstact class {cls}")
