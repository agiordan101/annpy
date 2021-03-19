
def parse_object(obj, cls, data, default_cls):
	return obj() if isinstance(obj, cls) else data.get(obj, default_cls)()
