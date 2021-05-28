# def load_model(file_path):

# 	model = None
# 	with open(file_path, 'r') as f:
# 		data = json.loads(f.read())
		
# 		model = data.get('file_type')
# 		if model != "Only weights":
# 			raise Exception(f"[annpy error] load_model: Wrong <file_type> for file {file_path}")

# 		print(f"MODEL FILE DATA:\n{data}")

# 	return model