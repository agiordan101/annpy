def load_model(file_path):

	with open(file_path, 'r') as f:
		data = json.loads(f.read())
		
		if data.get('file') != self.save_weights_method:
			print(f"Wrong format for file {file_path}")
			return
		
		model = data['file']

	else:
		raise Exception(f"[annpy error] Unable to open {file_path}")

