import sys
import annpy
from DataProcessing import DataProcessing

folder_path = "./ressources"
model_name = "loadtest"

def parsing(dataset_path, seeds_path=None):

	data = DataProcessing()
	data.parse_dataset(dataset_path=dataset_path,
						columns_range=[1, None],
						target_index=0)
	data.normalize()
	features, targets = data.get_data(binary_targets=['B', 'M'])

	seed = None
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
			
			print(f"end parsing, seed: {type(seed)}, loss: {best_loss}\n")

	except:
		print(f"No seed.\n")

	return features, targets, features[0].shape[0], seed

def get_my_model(input_shape):

	model = annpy.models.SequentialModel(
		input_shape=input_shape,
		name=model_name
	)
	model.add(annpy.layers.FullyConnected(10))
	model.add(annpy.layers.FullyConnected(5))
	model.add(annpy.layers.FullyConnected(2, activation="Softmax"))
	model.compile()
	return model

# Protection & Parsing
if len(sys.argv) < 2:
	raise Exception("usage: python3 test.py dataset [seeds]")
else:
	print(f"dataset: {sys.argv[1]}\nseeds: {sys.argv[2] if len(sys.argv) > 2 else None}\n")

features, targets, input_shape, seed = parsing(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)


model = get_my_model(input_shape)
file_path, _ = model.save_weights(folder_path)

model = annpy.models.SequentialModel.load_model(file_path)

# print(model)
# exit(0)
model.compile(loss="BinaryCrossEntropy")
logs = model.fit(
	features,
	targets,
	callbacks=[
		annpy.callbacks.EarlyStopping(
			model=model,
			monitor='val_BinaryCrossEntropy',
			patience=10,
		)
	]
)
print(f"Fit result: {logs}")
