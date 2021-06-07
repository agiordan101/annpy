import sys
import json
import annpy
import numpy as np

loss = "BinaryCrossEntropy"
monitored_loss = f'val_{loss}'
n_seed_search = 42

def parsing(dataset_path, seeds_path=None):

	data = annpy.parsing.DataProcessing()
	data.parse_dataset(dataset_path=dataset_path,
						columns_range=[1, None],
						target_index=0)
	data.normalize()
	features, targets = data.get_data(binary_targets=['B', 'M'])

	seed = None
	try:
		with open(seeds_path, 'r') as f:
			lines = [elem for elem in f.read().split('\n') if elem and elem[0] == '{']
			
			global best_loss_file
			best_loss_file = 42
			for line in lines:

				# print(f"line {type(line)}: {line}")
				line = json.loads(line)
				if line.get(monitored_loss, None) < best_loss_file:
					best_loss_file = line.get(monitored_loss, None)
					seed = line.get('seed', None)

			print(f"end parsing, seed: {type(seed)}, loss: {best_loss_file}\n")

	except:
		print(f"No seed.\n")

	return features, targets, features[0].shape[0], seed

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

	early_stopping = annpy.callbacks.EarlyStopping(
		model=model,
		monitor=monitored_loss,
		patience=15,
	)

	logs = model.fit(
		features,
		targets,
		epochs=500,
		batch_size=32,
		callbacks=[early_stopping],
		verbose=False,
		print_graph=graph
	)

	print(f"Fit result: {logs[monitored_loss][-1]}")
	logs = early_stopping.get_best_metrics()
	print(f"Fit best  : {logs}")
	return model, logs


def estimate_tts_seed(input_shape, seed, tts_seed, iter=5):
	# return sum(get_model_train(input_shape, seed)[1][monitored_loss] for i in range(iter)) / iter
	losses = 0
	for i in range(iter):
		print(f"\nTrain {i+1}/{iter} ...")
		losses += get_model_train(input_shape, seed, tts_seed)[1][monitored_loss]
	return losses / iter


# Protection
if len(sys.argv) < 3:
	raise Exception("usage: python3 test.py dataset seeds")
else:
	print(f"dataset: {sys.argv[1]}\nseeds: {sys.argv[2] if len(sys.argv) > 2 else None}\n")

# Parsing
features, targets, input_shape, seed = parsing(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)

layers_shp = (input_shape, 64, 32, 2)

# mode = get_model(input_shape, seed)
# mode2 = get_model(input_shape, seed)

# mode.deepsummary()
# mode2.deepsummary()
# np.random.seed(42)
# print(np.random.get_state())
# np.random.seed(101)
# print(np.random.get_state())
# i=42
# np.random.seed(i)
# print(np.random.get_state())
# exit(0)

best_tts_seed = None
best_loss = 1
for i in range(n_seed_search):

	tts_seed = int(np.random.randn()) % 1000000
	np.random.seed(tts_seed)
	loss_ = estimate_tts_seed(input_shape, seed, np.random.get_state())

	if loss_ < best_loss:
		best_loss = loss_
		best_tts_seed = tts_seed

	print(f"tts_seed {i}: {loss_}\n")

print(f"\tBEST TTS_SEED ({n_seed_search}): {best_tts_seed} -- loss: {loss_}\n")

if best_loss < best_loss_file:

	np.random.seed(best_tts_seed)
	tts_seed = np.random.get_state()
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

else:
	print(f"Best loss found is worst than best loss file fetch in seeds file: {best_loss} (script) > (file) {best_loss_file}")