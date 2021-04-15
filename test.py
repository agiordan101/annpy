# from annpy.models.Sequencial import Sequencial
import sys
import annpy
from DataProcessing import DataProcessing
import numpy as np

def get_model():

	model = annpy.models.Sequencial(input_shape=29, name="First model")

	model.add(annpy.layers.FullyConnected(
		256,
		activation="ReLU",
	))
	model.add(annpy.layers.FullyConnected(
		128,
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
		# optimizer="Adam",
		optimizer=annpy.optimizers.Adam(
<<<<<<< HEAD
			lr=0.002
=======
			lr=0.0003
>>>>>>> dd22d716030f651bbbc6e66826c6dc4804f60a95
		),
		# optimizer=annpy.optimizers.RMSProp(
		# 	lr=0.0001,
		# 	momentum=0.90,
		# ),
		# optimizer=annpy.optimizers.SGD(
		# 	lr=0.1,
		# 	# momentum=0.92,
		# ),
		# metrics=[
		# 	# "MSE",
		# 	annpy.metrics.RangeAccuracy([0.5, 0.5])
		# ]
		metrics=["RangeAccuracy"]
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
	batch_size=30,
	callbacks=[
		annpy.callbacks.EarlyStopping(
			monitor='val_BinaryCrossEntropy',
			patience=500,
		)
	],
	# val_percent=None, # Bug
	verbose=False
)

# loss, accuracy = model.fit(
# 	features,
# 	targets,
# 	epochs=7,
# 	verbose=True
# )
