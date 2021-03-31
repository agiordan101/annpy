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
		50,
		activation="ReLU",
	))
	model.add(annpy.layers.FullyConnected(
		30,
		activation="ReLU",
	))
	model.add(annpy.layers.FullyConnected(
		15,
		activation="ReLU",
	))
	model.add(annpy.layers.FullyConnected(
		2,
		activation="Sigmoid",
	))
	model.compile(
		loss="MSE",
		# optimizer="SGD",
		optimizer=annpy.optimizers.SGD(0.01),
		metrics=[annpy.metrics.RangeAccuracy([0.5, 0.5])]
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

model = get_model()
model.summary()
# model.deepsummary()

loss, accuracy = model.fit(
	features,
	targets,
	epochs=500,
	batch_size=100,
	callbacks=[
		annpy.callbacks.EarlyStopping(
			monitor='val_MSE',
			patience=5,
			min_delta=0.0001,
			mode='min'
		)
	],
	val_percent=None,
	verbose=True
)

# loss, accuracy = model.fit(
# 	features,
# 	targets,
# 	epochs=7,
# 	verbose=True
# )
