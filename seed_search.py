import sys
import json
import annpy
import numpy as np

from DataProcessing import DataProcessing

loss = "BinaryCrossEntropy"
monitored_loss = f'val_{loss}'
n_seed_search = 42

def parsing(dataset_path, seeds_path=None):

	data = DataProcessing()
	data.parse_dataset(dataset_path=dataset_path,
						columns_range=[1, None],
						target_index=0)
	data.normalize()
	features, targets = data.get_data(binary_targets=['B', 'M'])

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
					tts_seed_ = line.get('tts_seed', None)

			print(f"end parsing, seed: {type(seed)}, loss: {best_loss}\n")

	except:
		print(f"No seed.\n")

	return features, targets, features[0].shape[0], tts_seed_

def get_model(input_shape, seed=None):

	model = annpy.models.SequentialModel(
		input_shape=layers_shp[0],
		name="First model",
		seed=seed,
		tts_seed=tts_seed
	)
	# model = annpy.models.sequential(input_shape=input_shape, name="First model")

	model.add(annpy.layers.FullyConnected(
		layers_shp[1],
		activation="relu",
	))
	model.add(annpy.layers.FullyConnected(
		layers_shp[2],
		# activation="relu",
		activation="tanh",
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

def get_model_train(input_shape, seed=None, graph=False):

	model = get_model(input_shape, seed)
	logs = model.fit(
		features,
		targets,
		epochs=100,
		batch_size=32,
		callbacks=[
			annpy.callbacks.EarlyStopping(
				model=model,
				monitor=monitored_loss,
				patience=5,
			)
		],
		# val_percent=None, # Bug
		verbose=False,
		print_graph=graph
	)
	print(f"Fit result: {logs}")
	return model, logs

def estimate_seed(input_shape, seed, iter=5):
	# return sum(get_model_train(input_shape, seed)[1][monitored_loss] for i in range(iter)) / iter
	losses = 0
	for i in range(iter):
		print(f"Train {i+1}/{iter} ...")
		losses += get_model_train(input_shape, seed)[1][monitored_loss]
	return losses / iter


# Protection & Parsing
if len(sys.argv) < 2:
	raise Exception("usage: python3 test.py dataset [seeds]")
else:
	print(f"dataset: {sys.argv[1]}\nseeds: {sys.argv[2] if len(sys.argv) > 2 else None}\n")

features, targets, input_shape, tts_seed = parsing(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
layers_shp = (input_shape, 64, 32, 2)


# Model search
best_model = None
best_loss = 42

i = 0
while i < n_seed_search:

	print(f"-- Seed {i+1}/{n_seed_search} --")
	model, logs = get_model_train(input_shape)
	print()

	seed_loss = estimate_seed(input_shape, model.get_seed())

	if seed_loss < best_loss:
		best_loss = seed_loss
		best_model = model

	print(f"Seed {i+1}/{n_seed_search} -- Average loss: {seed_loss}\t-- Best loss: {best_loss}\n")
	i += 1

print(f"Average loss of the best seed ({n_seed_search} tries): {best_loss}")


seed = best_model.get_seed()
with open("ressources/seeds.txt", 'a') as f:
	seed = list(seed)
	tts_seed = list(tts_seed)

	seed[1] = [int(n) for n in seed[1]]
	tts_seed[1] = [int(n) for n in tts_seed[1]]

	seed_dict = {
		monitored_loss: best_loss,
		'layers_shp': layers_shp,
		'seed': seed,
		'tts_seed': tts_seed,
	}
	print(f"seed == tts_seed -> {seed == tts_seed}")
	print(f"Write best loss in file")
	json.dump(seed_dict, f)
	print('\n', file=f)
