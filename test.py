# from annpy.models.Sequencial import Sequencial
import annpy
import numpy as np
from annpy.optimizers.Optimizer import Optimizer

# XOR test
# inputs = np.array([[0, 0]])
# targets = np.array([[0]])

# inputs = np.array([[0, 0],
# 					[0, 1],
# 					[1, 0]])
# targets = np.array([[0],
# 					[1],
# 					[1]])

inputs = np.array([[0, 0],
					[0, 1],
					[1, 0],
					[1, 1]])
targets = np.array([[0],
					[1],
					[1],
					[0]])

model = annpy.models.Sequencial(input_shape=2, name="First model")
model.add(annpy.layers.FullyConnected(4, activation="Sigmoid"))
model.add(annpy.layers.FullyConnected(1, activation="Sigmoid"))
model.compile(loss="MSE",
				optimizer=annpy.optimizers.SGD(20),
				metrics=["Accuracy"])
model.deepsummary()

# output = [[1, 2, 3, 4, 5], [2, 2, 3, 3.9, 5.1]]
# target = [[1, 2, 3, 4, 5], [2, 2, 3, 3.9, 5.1]]

# metric = annpy.metrics.Accuracy()
# metric(output, target)
# output = [[1, 2, 3, 4, 5]]
# target = [[1, 2, 3, 2, 5]]
# metric(output, target)
# print(metric.get_result())

# exit(0)


prediction = model.forward(np.array(inputs))
print(f"PREDICTION ->\nInputs {len(inputs)}: {inputs}\nOutputs {len(prediction)}: {prediction}\nOutputs {len(targets)}: {targets}\n")

loss = model.fit(inputs,
					targets,
					epochs=42,
					batch_size=4,
					verbose=False)
print(f"Model loss: {loss}")

prediction = model.forward(np.array(inputs))
print(f"PREDICTION ->\nInputs {len(inputs)}: {inputs}\nOutputs {len(prediction)}: {prediction}\nOutputs {len(targets)}: {targets}\n\n")
