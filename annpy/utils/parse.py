import annpy
from annpy.losses.MSE import MSE
from annpy.losses.BinaryCrossEntropy import BinaryCrossEntropy
from annpy.optimizers.SGD import SGD
from annpy.optimizers.Adam import Adam
from annpy.optimizers.RMSProp import RMSProp
from annpy.metrics.Accuracy import Accuracy
from annpy.metrics.RangeAccuracy import RangeAccuracy
from annpy.activations.ReLU import ReLU
from annpy.activations.Linear import Linear
from annpy.activations.Sigmoid import Sigmoid
from annpy.activations.Softmax import Softmax
from annpy.initializers.LecunNormal import LecunNormal
from annpy.initializers.LecunUniform import LecunUniform
from annpy.initializers.GlorotNormal import GlorotNormal
from annpy.initializers.GlorotUniform import GlorotUniform
from annpy.initializers.RandomNormal import RandomNormal
from annpy.initializers.RandomUniform import RandomUniform
from annpy.initializers.Ones import Ones
from annpy.initializers.Zeros import Zeros

# Only objects that don't need arguments in __init__() can be past
# If not, Object need to be past, not String
objects = {
	'mse': MSE,
	'sgd': SGD,
	'relu': ReLU,
	'ones': Ones,
	'adam': Adam,
	'zeros': Zeros,
	'linear': Linear,
	'sigmoid': Sigmoid,
	'softmax': Softmax,
	'rmsprop': RMSProp,
	'accuracy': Accuracy,
	'lecunnormal': LecunNormal,
	'lecununiform': LecunUniform,
	'glorotnormal': GlorotNormal,
	'glorotuniform': GlorotUniform,
	'randomnormal': RandomNormal,
	'randomuniform': RandomUniform,
	'rangeaccuracy': RangeAccuracy,
	'binarycrossentropy': BinaryCrossEntropy,
}

def parse_object(obj, cls, str_allowed=True):

	print(f"obj={obj}")
	print(f"cls={cls}")
	if str_allowed and isinstance(obj, str) and obj.lower() in objects:
		obj = objects[obj.lower()]()

	if issubclass(type(obj), cls):
		return obj
	else:
		raise Exception(f"Object {obj} (type: {type(obj)}) is not a child of abstact class {cls}")
