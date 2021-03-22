import annpy

objects = {
	'MSE':		annpy.losses.MSE,
	'SGD':		annpy.optimizers.SGD,
	'ReLU':		annpy.activations.ReLU,
	'Linear':	annpy.activations.Linear,
	'Sigmoid':	annpy.activations.Sigmoid,
}

def parse_object(obj, cls, str_allowed=True):

	if str_allowed and isinstance(obj, str) and obj in objects:
		obj = objects[obj]()

	if issubclass(type(obj), cls):
		return obj
	else:
		raise Exception(f"Object {obj} is not a child of abstact class {cls}")
