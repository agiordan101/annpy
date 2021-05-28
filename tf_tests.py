import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import annpy
from DataProcessing import DataProcessing

def parsing():

	data = DataProcessing()
	data.parse_dataset(dataset_path="ressources/data.csv",
						columns_range=[1, None],
						target_index=0)
	data.normalize()
	data = data.get_data(binary_targets=['B', 'M'])
	return data

def get_my_model(input_shape):

	model = annpy.models.SequentialModel(input_shape=input_shape)
	model.add(annpy.layers.FullyConnected(10))
	model.add(annpy.layers.FullyConnected(5))
	model.add(annpy.layers.FullyConnected(2, activation="Softmax"))
	model.compile(loss="mse", optimizer='sgd')
	return model

def get_tf_model(input_shape):

	inputs = tf.keras.Input(shape=(input_shape,))
	d1 = tf.keras.layers.Dense(10)(inputs)
	d2 = tf.keras.layers.Dense(5)(d1)
	outputs = tf.keras.layers.Dense(2, activation=tf.nn.softmax)(d2)
	model = tf.keras.Model(inputs=inputs, outputs=outputs)
	model.compile(loss="mse", optimizer='sgd')
	return model



features, targets = parsing()
input_shape = features[0].shape[0]

tf.random.set_seed(42)
np.random.seed(42)

my_model = get_my_model(input_shape)
tf_model = get_tf_model(input_shape)

tf_model.layers[1].set_weights(my_model.weights[0])
tf_model.layers[2].set_weights(my_model.weights[1])
tf_model.layers[3].set_weights(my_model.weights[2])
# tf_model.set_weights(my_model.weights)

my_model.deepsummary()
print([l.kernel for l in tf_model.layers])

logs = my_model.fit(
	features,
	targets,
	epochs=1,
	batch_size=568,
	val_percent=0, # Bug
	shuffle=False,
	verbose=True,
	print_graph=False
)

logs = tf_model.fit(
	features,
	targets,
	epochs=1,
	batch_size=568,
	validation_split=0,
	shuffle=False
)

# print(my_model.forward(features[0]))
# print(tf_model.forward(features[0]))

my_model.deepsummary()
print([l.kernel for l in tf_model.layers])