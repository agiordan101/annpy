import annpy

def get_my_model(input_shape):

	model = annpy.models.SequencialModel(input_shape=input_shape)
	model.add(annpy.layers.FullyConnected(10))
	model.add(annpy.layers.FullyConnected(5))
	model.add(annpy.layers.FullyConnected(2, activation="Softmax"))
	model.compile(loss="mse", optimizer='sgd')
	return model

model = get_my_model(20)

model.save_weights("./ressources")
