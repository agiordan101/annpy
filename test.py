# from annpy.models.Sequencial import Sequencial
import sys
import annpy
from DataProcessing import DataProcessing
import numpy as np

# def get_model():

# 	model = annpy.models.Sequencial(input_shape=29, name="First model")
# 	model.add(annpy.layers.FullyConnected(
# 		40,
# 		activation="ReLU",
# 		kernel_initializer='LecunUniform',
# 		bias_initializer='Zeros'
# 	))
# 	model.add(annpy.layers.FullyConnected(
# 		25,
# 		activation="ReLU",
# 		kernel_initializer='LecunNormal',
# 		bias_initializer='Ones'
# 	))
# 	model.add(annpy.layers.FullyConnected(
# 		10,
# 		activation="ReLU",
# 		kernel_initializer='RandomUniform',
# 		bias_initializer='RandomNormal'
# 	))
# 	model.add(annpy.layers.FullyConnected(
# 		2,
# 		activation="Sigmoid",
# 		kernel_initializer='GlorotUniform',
# 		bias_initializer='GlorotNormal'
# 	))
# 	model.compile(
# 		loss="MSE",
# 		optimizer="SGD",
# 		# optimizer=annpy.optimizers.SGD(),
# 		metrics=[annpy.metrics.RangeAccuracy([0.5, 0.5])]
# 	)
# 	return model
def get_model():

	model = annpy.models.Sequencial(input_shape=29, name="First model")

	model.add(annpy.layers.FullyConnected(
		40,
		activation="ReLU",
	))
	model.add(annpy.layers.FullyConnected(
		20,
		activation="ReLU",
	))
	model.add(annpy.layers.FullyConnected(
		10,
		activation="ReLU",
	))
	model.add(annpy.layers.FullyConnected(
		2,
		# activation="Sigmoid",
		activation="Softmax",
	))
	model.compile(
		loss="BinaryCrossEntropy",
		# loss="MSE",
		optimizer=annpy.optimizers.SGD(0.1),
		metrics=[
			# "MSE",
			annpy.metrics.RangeAccuracy([0.5, 0.5])
		]
	)
	return model

if len(sys.argv) < 2:
	raise Exception("usage: python3 test.py dataset")

data = DataProcessing()
# data.load_data("ressources/normalization.txt", normalization=True)
data.parse_dataset(dataset_path="ressources/data.csv",
					columns_range=[1, -1],
					target_index=0)
data.normalize()
features, targets = data.get_data(binary_targets=['B', 'M'])

# print([f for f in np.nditer(features)])
# exit(0)

# print(targets)
# first = len([1 for m, b in targets if m == 1.])
# print(first)
# print(len(targets) - first)
# exit(0)

model = get_model()
model.summary()
# model.deepsummary()

# softmax = annpy.activations.Softmax()
# a = np.array([[1, 2, 1]])
# print(softmax(a))

loss, accuracy = model.fit(
	features,
	targets,
	epochs=500,
	batch_size=42,
	callbacks=[
		# annpy.callbacks.EarlyStopping(
		# 	monitor='val_RangeAccuracy',
		# 	patience=10,
		# ),
		annpy.callbacks.EarlyStopping(
			monitor='val_BinaryCrossEntropy',
			# monitor='val_RangeAccuracy',
			# monitor='BinaryCrossEntropy',
			# monitor='val_MSE',
			patience=20,
		)
	],
	# val_percent=None,
	verbose=False
)

# loss, accuracy = model.fit(
# 	features,
# 	targets,
# 	epochs=7,
# 	verbose=True
# )
