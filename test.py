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
		10,
		activation="linear",
	))
	model.add(annpy.layers.FullyConnected(
		5,
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
		metrics=["RangeAccuracy"]
	)
	return model

if len(sys.argv) < 2:
	raise Exception("usage: python3 test.py dataset")

data = DataProcessing()
# data.load_data("ressources/normalization.txt", normalization=True)
data.parse_dataset(dataset_path="ressources/data.csv",
					columns_range=[1, None],
					target_index=0)
data.normalize()
features, targets = data.get_data(binary_targets=['B', 'M'])

model = get_model(features[0].shape[0])
# model.summary()
# model.deepsummary()

logs0 = model.fit(
	features,
	targets,
	epochs=300,
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
	# print_graph=False
)

seed = model.get_seed()
model = get_model(features[0].shape[0], seed=seed)
# model.deepsummary()

logs1 = model.fit(
	features,
	targets,
	epochs=300,
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
	# print_graph=False
)

print(f"Logs model 0: {logs0}")
print(f"Logs model 1: {logs1}")

# init = annpy.initializers.UniformInitializer(min_val=-10, max_val=2.3)
# print(init((3, 8)))
