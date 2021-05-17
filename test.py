import sys
import annpy
from DataProcessing import DataProcessing
import numpy as np

def get_model(input_shape, seed=None):

	model = annpy.models.SequencialModel(
		input_shape=input_shape,
		name="First model",
		seed=seed
	)
	# model = annpy.models.Sequencial(input_shape=input_shape, name="First model")

	model.add(annpy.layers.FullyConnected(
		128,
		activation="linear",
	))
	model.add(annpy.layers.FullyConnected(
		64,
		activation="linear",
		# activation="tanh",
	))
	model.add(annpy.layers.FullyConnected(
		2,
		# activation="Sigmoid",
		activation="Softmax",
	))
	model.compile(
		loss="BinaryCrossEntropy",
		# loss="MSE",
		# optimizer="SGD",
		optimizer="Adam",
		# optimizer=annpy.optimizers.Adam(
		# 	lr=0.001
		# ),
		# optimizer=annpy.optimizers.SGD(
		# 	lr=0.2,
		# 	momentum=0.92,
		# ),
		# metrics=["RangeAccuracy"]
	)
	return model

def get_trainned_model(input_shape, seed=None):

	model = get_model(input_shape, seed)
	logs = model.fit(
		features,
		targets,
		epochs=100,
		batch_size=32,
		callbacks=[
			annpy.callbacks.EarlyStopping(
				model=model,
				monitor='val_BinaryCrossEntropy',
				patience=10,
			)
		],
		# val_percent=None, # Bug
		verbose=False,
		print_graph=False
	)
	return model, logs



if len(sys.argv) < 2:
	raise Exception("usage: python3 test.py dataset")

data = DataProcessing()
# data.load_data("ressources/normalization.txt", normalization=True)
data.parse_dataset(dataset_path="ressources/data.csv",
					columns_range=[1, None],
					target_index=0)
data.normalize()
features, targets = data.get_data(binary_targets=['B', 'M'])

logs = {'loss': 1}

# Seed search
best_seed = None
best_loss = 42
i = 0
s = 0
while logs['loss'] > 0.01 and i < 100:

	# model, logs = get_trainned_model(features[0].shape[0], seed=best_seed)
	model, logs = get_trainned_model(features[0].shape[0])
	
	if logs['loss'] < best_loss:
		best_loss = logs['loss']
		best_seed = model.get_seed()
	
	i += 1
	s += logs['loss']
	print(f"{i} -- Best loss: {best_loss} -- Average: {s / i} \tloss: {logs['loss']}")

# i = 0
# while i < 10:
# 	model, logs = get_trainned_model(features[0].shape[0], seed=best_seed)
# 	print(f"best loss with this seed: {logs['loss']} --\tBest loss: {best_loss}")
# 	i += 1

