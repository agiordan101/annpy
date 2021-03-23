# from annpy.models.Sequencial import Sequencial
import annpy
import numpy as np
from annpy.optimizers.Optimizer import Optimizer

model = annpy.models.Sequencial(input_shape=2, name="First model")
model.add(annpy.layers.FullyConnected(4, activation="Sigmoid"))
model.add(annpy.layers.FullyConnected(1, activation="Sigmoid"))
model.compile(loss="MSE",
				optimizer=annpy.optimizers.SGD(1))
model.deepsummary()

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

prediction = model.forward(np.array(inputs))
print(f"PREDICTION ->\nInputs {len(inputs)}: {inputs}\nOutputs {len(prediction)}: {prediction}\nOutputs {len(targets)}: {targets}\n\n")

loss = model.fit(inputs, targets, epochs=42000, batch_size=4, verbose=False)
print(f"Model loss: {loss}")

prediction = model.forward(np.array(inputs))
print(f"PREDICTION ->\nInputs {len(inputs)}: {inputs}\nOutputs {len(prediction)}: {prediction}\nOutputs {len(targets)}: {targets}\n\n")
