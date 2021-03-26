# from annpy.models.Sequencial import Sequencial
import sys
import annpy
from DataProcessing import DataProcessing
import numpy as np

def get_model():
	model = annpy.models.Sequencial(input_shape=29, name="First model")
	model.add(annpy.layers.FullyConnected(60, activation="Sigmoid"))
	model.add(annpy.layers.FullyConnected(30, activation="Sigmoid"))
	model.add(annpy.layers.FullyConnected(2, activation="Sigmoid"))
	model.compile(loss="MSE",
					optimizer=annpy.optimizers.SGD(1),
					metrics=[annpy.metrics.RangeAccuracy([0.5, 0.5])])
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

print(f"features normalized:\n{type(features)} / {len(features[0])}")
print(f"targets normalized:\n{type(targets)} / {len(targets)}")

model = get_model()
model.deepsummary()

loss, accuracy = model.fit(features,
					targets,
					epochs=420,
					batch_size=42,
					verbose=True)


# prediction = model.forward(np.array(inputs))
# print(f"PREDICTION ->\nInputs {len(inputs)}: {inputs}\nOutputs {len(prediction)}: {prediction}\nOutputs {len(targets)}: {targets}\n")

# XOR test
# inputs = np.array([[0, 0]])
# targets = np.array([[0]])

# inputs = np.array([[0, 0],
# 					[0, 1],
# 					[1, 0]])
# targets = np.array([[0],
# 					[1],
# 					[1]])