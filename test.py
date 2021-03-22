# from annpy.models.Sequencial import Sequencial
import annpy
import numpy as np
from annpy.optimizers.Optimizer import Optimizer

model = annpy.models.Sequencial(input_shape=2, name="First model")
model.add(annpy.layers.FullyConnected(4, activation="Sigmoid"))
model.add(annpy.layers.FullyConnected(1, activation="Sigmoid"))
model.compile()
model.deepsummary()

# XOR test
inputs = np.array([[0, 0],
					[0, 1],
					[1, 0],
					[1, 1]])
targets = np.array([[0],
					[1],
					[1],
					[0]])

print(f"Inputs: {inputs}\nOutputs: {model.forward(np.array(inputs))}")

model.fit(inputs, targets)
# print(f"Inputs: {inputs}\nOutputs: {model.forward(np.array(inputs))}")
