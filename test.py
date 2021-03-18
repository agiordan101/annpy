from MLP_lib import *

model = Model(input_shape=2, name="First model")
model.add(Layer(4, activation=fa.ReLU))
model.add(Layer(1, activation=fa.ReLU))
model.compile()
# model.deepsummary()

# XOR test
inputs = np.array([[0, 0],
					[0, 1],
					[1, 0],
					[1, 1]])
targets = np.array([[0],
					[1],
					[1],
					[0]])

for input_ in inputs:
	print(f"{input_} -> {model.forward(np.array(input_))}")

# model.fit()