import numpy as np
from annpy.callbacks.Callback import Callback

class EarlyStopping(Callback):

	def __init__(self,
					model,
					monitor='val_loss',
					min_delta=0,
					patience=0,
					mode='auto'):
		super().__init__(model)

		if mode not in ['auto', 'min', 'max']:
			raise Exception(f"Can't resolve argument mode={mode} in EarlyStopping constructor")

		self.monitor = monitor
		self.min_delta = min_delta
		self.patience = patience
		self.mode = mode

		# if mode == 'auto':
		# 	if 'loss' in monitor.lower():
		# 		mode = 'min'
		# 	elif 'accuracy' in monitor.lower():
		# 		mode = 'max'
		# 	else:

	def on_train_begin(self, **kwargs):

		if self.monitor in self.model.metrics:

			self.best_val = np.inf
			self.fails = 0

			if self.mode == 'auto':
				self.mode = self.model.metrics[self.monitor].get_variation_goal()
			
			self.sign = 1 if self.mode == 'min' else -1

		else:
			print(f"Metrics:\n{model.metrics}")
			raise Exception(f"Metric argument in EarlyStopping constructor isn't exist: '{self.monitor}'")

	def on_epoch_begin(self, **kwargs):
		pass

	def on_batch_begin(self, **kwargs):
		pass
	
	def on_batch_end(self, **kwargs):
		pass
	
	def on_epoch_end(self, verbose=True, **kwargs):

		# value = self.model.current_metrics[self.monitor].get_result() * self.sign
		value = self.model.metrics[self.monitor].get_result() * self.sign

		# print(f"Mode={self.mode}: {value} < {self.best_val} - {self.min_delta}")

		if value <= self.best_val - self.min_delta:
			self.best_val = value
			self.fails = 0

		else:
			# if value > self.best_val + 0.5:
			# 	print(f"WTTF bestval:{self.best_val}\tvalue:{value}")
			# 	exit(0)
			# print(f"FAIL {self.monitor} :{abs(self.best_val)}\tvalue:{value}")

			self.fails += 1
			if self.fails > self.patience:
				if verbose:
					print(f"----------------------")
					self.summary()
					print(f"{self.monitor} -> on_epoch_end -> Stop trainning")
					print(f"No improvement after {self.patience} epochs")
					print(f"Best {self.monitor}: {abs(self.best_val)}\n")
					print(f"----------------------")
				self.model.stop_trainning = True

	def on_train_end(self, **kwargs):
		pass

	def summary(self):
		print(f"Callbacks:\tannpy.callbacks.EarlyStopping")
