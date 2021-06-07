from annpy.losses.MSE import MSE
from annpy.losses.BinaryCrossEntropy import BinaryCrossEntropy

from annpy.optimizers.SGD import SGD
from annpy.optimizers.Adam import Adam
from annpy.optimizers.RMSProp import RMSProp

from annpy.metrics.Accuracy import Accuracy
from annpy.metrics.RangeAccuracy import RangeAccuracy

from annpy.activations.ReLU import ReLU
from annpy.activations.Tanh import Tanh
from annpy.activations.Linear import Linear
from annpy.activations.Sigmoid import Sigmoid
from annpy.activations.Softmax import Softmax

from annpy.initializers.Ones import Ones
from annpy.initializers.Zeros import Zeros
from annpy.initializers.LecunNormal import LecunNormal
from annpy.initializers.LecunUniform import LecunUniform
from annpy.initializers.GlorotNormal import GlorotNormal
from annpy.initializers.RandomNormal import RandomNormal
from annpy.initializers.GlorotUniform import GlorotUniform
from annpy.initializers.RandomUniform import RandomUniform

from annpy.layers.FullyConnected import FullyConnected

# Only objects that don't need arguments in __init__() can be past
# If not, Object need to be past, not String
objects = {
	'mse': MSE,
	'sgd': SGD,
	'relu': ReLU,
	'tanh': Tanh,
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
	'randomnormal': RandomNormal,
	'glorotuniform': GlorotUniform,
	'randomuniform': RandomUniform,
	'rangeaccuracy': RangeAccuracy,
	'binarycrossentropy': BinaryCrossEntropy,
	'fullyconnected': FullyConnected,
}

def parse_object(obj, cls, instantiation=True, **kwargs):
	"""
		Can convert String to Object
		Can convert String to Class
		Check if result object is an instance of cls
	"""

	obj_cls = None
	if isinstance(obj, str):

		obj_cls = objects.get(obj.lower())
		if not obj_cls:
			raise Exception(f"[annpy error] parse_object: str {obj} unrecognized.")

		obj = obj_cls(**kwargs) if instantiation else obj_cls
	
	# print(f"Obj {obj}\nobj_cls {obj_cls}\ncls needed {cls}\n")

	if issubclass(type(obj), cls):
		return obj
	else:
		raise Exception(f"[annpy error] parse_object: Object {obj} (type: {type(obj)}) is not a child of abstact class {cls}")
