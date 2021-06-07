import sys
import json
import annpy
import numpy as np

from DataProcessing import DataProcessing

loss = "BinaryCrossEntropy"
monitored_loss = f'val_{loss}'
n_seed_search = 2

def parsing(dataset_path, seeds_path=None):

	data = DataProcessing()
	data.parse_dataset(dataset_path=dataset_path,
						columns_range=[1, None],
						target_index=0)
	data.normalize()
	features, targets = data.get_data(binary_targets=['B', 'M'])

	seed = None
	tts_seed_ = None
	try:
		with open(seeds_path, 'r') as f:
			lines = [elem for elem in f.read().split('\n') if elem and elem[0] == '{']
			
			best_loss = 42
			for line in lines:

				# print(f"line {type(line)}: {line}")
				line = json.loads(line)
				if line.get(monitored_loss, None) < best_loss:
					best_loss = line.get(monitored_loss, None)
					seed = line.get('seed', None)
					tts_seed_ = line.get('tts_seed', None)
			
			print(f"end parsing, seed: {type(seed)}, loss: {best_loss}\n")

	except:
		print(f"No seed.\n")

	return features, targets, features[0].shape[0], seed, tts_seed_

def get_model(input_shape, seed=None, tts_seed=None):

	model = annpy.models.SequentialModel(
		input_shape=layers_shp[0],
		name="First model",
		seed=seed,
		tts_seed=tts_seed
	)
	# model = annpy.models.sequential(input_shape=input_shape, name="First model")

	model.add(annpy.layers.FullyConnected(
		layers_shp[1],
		# activation="relu",
	))
	model.add(annpy.layers.FullyConnected(
		layers_shp[2],
		# activation="relu",
		# activation="tanh",
	))
	model.add(annpy.layers.FullyConnected(
		layers_shp[3],
		# activation="Sigmoid",
		activation="Softmax",
	))
	model.compile(
		loss=loss,
		# loss="MSE",
		# optimizer="SGD",
		optimizer="Adam",
		# optimizer=annpy.optimizers.Adam(
		# 	lr=0.001
		# ),
		# optimizer=annpy.optimizers.SGD(
		# 	lr=0.2,
		# 	momentum=0.92,
		# ),
		# metrics=["RangeAccuracy"]
	)
	return model

def get_model_train(input_shape, seed=None, tts_seed=None, graph=False):

	model = get_model(input_shape, seed, tts_seed)
	logs = model.fit(
		features,
		targets,
		epochs=100,
		batch_size=32,
		callbacks=[
			annpy.callbacks.EarlyStopping(
				model=model,
				monitor=monitored_loss,
				patience=15,
			)
		],
		# val_percent=None, # Bug
		verbose=False,
		print_graph=graph
	)
	print(f"Fit result: {logs}")
	return model, logs

def estimate_tts_seed(input_shape, seed, iter=3):
	# return sum(get_model_train(input_shape, seed)[1][monitored_loss] for i in range(iter)) / iter
	losses = 0
	for i in range(iter):
		print(f"Train {i+1}/{iter} ...")
		losses += get_model_train(input_shape, seed)[1][monitored_loss]
	return losses / iter



# Protection
if len(sys.argv) < 2:
	raise Exception("usage: python3 test.py dataset [seeds]")
else:
	print(f"dataset: {sys.argv[1]}\nseeds: {sys.argv[2] if len(sys.argv) > 2 else None}\n")

# Parsing
tts_seed = np.random.get_state() # Random seed
features, targets, input_shape, seed, tts_seed_ = parsing(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)

layers_shp = (input_shape, 64, 32, 2)
if tts_seed_:
	tts_seed = tts_seed_

for i in range(4):

	mode = annpy.models.
