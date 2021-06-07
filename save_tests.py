import sys
import annpy
import json
from DataProcessing import DataProcessing

folder_path = "./ressources"
model_name = "loadtest"
loss = 'BinaryCrossEntropy'
monitored_loss = f'val_{loss}'

def parsing(dataset_path, seeds_path=None):

	data = DataProcessing()
	data.parse_dataset(dataset_path=dataset_path,
						columns_range=[1, None],
						target_index=0)
	data.normalize()
	features, targets = data.get_data(binary_targets=['B', 'M'])

	seed = None
	tts_seed = None
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
					tts_seed = line.get('tts_seed', None)
			
			print(f"end parsing, seed: {type(seed)}, loss: {best_loss}\n")

	except:
		print(f"No seed.\n")

	return features, targets, features[0].shape[0], seed, tts_seed

def get_my_model(input_shape):

	model = annpy.models.SequentialModel(
		input_shape=layers_shp[0],
		name=model_name
	)
	model.add(annpy.layers.FullyConnected(layers_shp[1]))
	model.add(annpy.layers.FullyConnected(layers_shp[2]))
	model.add(annpy.layers.FullyConnected(layers_shp[3], activation="Softmax"))
	model.compile()
	return model

# Protection & Parsing
if len(sys.argv) < 2:
	raise Exception("usage: python3 test.py dataset [seeds]")
else:
	print(f"dataset: {sys.argv[1]}\nseeds: {sys.argv[2] if len(sys.argv) > 2 else None}\n")

features, targets, input_shape, seed, tts_seed = parsing(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)

layers_shp = [input_shape, 64, 18, 2]
model = get_my_model(input_shape)
file_path, _ = model.save_weights(folder_path)

model.summary()

model = annpy.models.SequentialModel.load_model(file_path)
model.compile(loss=loss)
model.summary()

layers_shp_load = [l.input_shape for l in model.sequence] + [model.sequence[-1].output_shape]
print(f"layers_shp: {layers_shp}\tlayers_shp_load: {layers_shp_load}")
if layers_shp_load != layers_shp:
	print(f"layers_shp are not equals: {layers_shp} != {layers_shp_load}")
	exit(0)

# print(model)
# exit(0)
logs = model.fit(
	features,
	targets,
	callbacks=[
		annpy.callbacks.EarlyStopping(
			model=model,
			monitor=monitored_loss,
			patience=10,
		)
	],
	verbose=False,
	print_graph=False
)
print(f"Fit result: {logs}")
