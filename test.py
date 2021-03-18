from MLP_lib import *

model = Model(input_shape=2, name="First model")
model.add(Layer(4, activation=fa.sigmoid))
model.add(Layer(1, activation=fa.sigmoid))
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

# model.fit()
# print(f"Inputs: {inputs}\nOutputs: {model.forward(np.array(inputs))}")
