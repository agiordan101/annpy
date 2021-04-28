import numpy as np

class DataProcessing():

	def __init__(self,
					features=None,
					targets=None,
					dataset_path=None,
					columns=None):

		if features and targets:

			if not isinstance(features, dict):
				raise Exception("Data parameter is not a instance Dict")
		# else:

			# if not dataset_path:
			# 	raise Exception("data or dataset_path arguments must be past")
			# self.parse_dataset(dataset_path)

		# print(f"features:\n{self.features}")
		# print(f"targets:\n{self.targets}")

		self.features = features
		self.targets = targets
		self.columns = columns
		self.normalization_data = []
		self.standardization_data = []

	def parse_dataset(self,
						dataset_path,
						columns_name=[],
						columns_range=[0, None],
						rows_range=[0, -1],
						parse_targets=True,
						target_index=1):

		#Open dataset file
		dataset_file = open(dataset_path, 'r')
		features_str = dataset_file.read()
		dataset_file.close()

		# Init data structure
		targets = []
		features = {}
		if columns_name:
			for feature in columns_name:
				features[feature] = []

		features_str_split = features_str.split("\n")[rows_range[0]:rows_range[1]] if rows_range[1] else features_str.split("\n")[rows_range[0]:]
		# Fill
		for student_str in features_str_split:
			student_strlst = student_str.split(',')[columns_range[0]:columns_range[1]] if columns_range[1] else student_str.split(',')[columns_range[0]:]

			if not features:
				for i in range(len(student_strlst) - (1 if parse_targets else 0)):
					features[f"feature_{i}"] = []

			if parse_targets:
				targets.append(student_strlst[target_index])
				student_strlst.pop(target_index)

			if columns_name:
				for i, feature in enumerate(columns_name):
					features[feature].append(float(student_strlst[i]) if student_strlst[i] else 0)
			else:
				for i, data in enumerate(student_strlst):
					features[f"feature_{i}"].append(float(data) if data else 0)

		self.features = features
		self.targets = targets
		return self.features, self.targets

	def normalize(self):

		# new_lst = []
		data = {}

		if self.normalization_data:
			
			for (feature, column), (_min, _max) in zip(self.features.items(), self.normalization_data):
				# new_lst.append([(x - data[0]) / (data[1] - data[0]) for x in item[1].values])
				data[feature] = [(x - _min) / (_max - _min) if isinstance(x, float) else x for x in column]

		else:
			for feature, column in self.features.items():
				_min = min(column)
				_max = max(column)
				# _min = column.min()
				# _max = column.max()
				self.normalization_data.append([_min, _max])
				data[feature] = [(x - _min) / (_max - _min) if isinstance(x, float) else x for x in column]
				# data[feature] = [(x - _min) / (_max - _min) for x in column.values]

		self.features = data
		# self.features = pd.DataFrame(data=data, columns=self.columns)

	def get_data(self, binary_targets=[]):

		features = np.array([np.array(features) for features in zip(*list(self.features.values()))])
		if binary_targets:
			targets = self.binary_targets_to_np(binary_targets[0], binary_targets[1])
		else:
			targets = np.array(self.targets)
		return features, targets

	def binary_targets_to_np(self, zero, one):

		targets = np.zeros((len(self.targets), 2))
		for i, label in enumerate(self.targets):
			# print(f"i={i} / label={label} / {zero} / {one}")
			if label == zero:
				targets[i] = np.array([0, 1])
			elif label == one:
				targets[i] = np.array([1, 0])
			else:
				targets[i] = np.nan
		return targets

	def save_data(self, file_path, normalization=False, standardization=False):

		with open(file_path, 'w') as f:

			if normalization:
				f.write("Normalization data\n")
				for _min, _max in self.normalization_data:
					f.write(f"{_min}/{_max}\n")
			
			if standardization:
				f.write("Standardization data\n")
				for mean, std in self.standardization_data:
					f.write(f"{mean}/{std}\n")

			f.close()

	def load_data(self, file_path, normalization=False, standardization=False):

		with open(file_path, 'r') as f:
			data = f.read()
			f.close()

			if normalization:
				self.normalization_data = [[float(x) for x in line.split('/')] for line in data.split('\n')[1:-1]]
				print(f"normalization_data: {self.normalization_data}")

			if standardization:
				self.standardization_data = [[float(x) for x in line.split('/')] for line in data.split('\n')[1:-1]]
				print(f"standardization_data: {self.standardization_data}")
