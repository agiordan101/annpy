import numpy as np
from annpy.callbacks.Callback import Callback

class EarlyStopping(Callback):

	def __init__(self,
					monitor='val_loss',
					min_delta=0,
					patience=0,
					mode='auto'):
		self.monitor = monitor
		self.min_delta = min_delta
		self.patience = patience

		if mode == 'auto':
			if 'loss' in monitor.lower():
				mode = 'min'
			elif 'accuracy' in monitor.lower():
				mode = 'max'
			else:
				raise Exception("Can't resolve auto mode parameter in EarlyStopping constructor")

		elif mode != 'min' and mode != 'max':
			raise Exception(f"Can't resolve argument mode={mode} in EarlyStopping constructor")

		self.mode = mode
		
		self.sign = 1 if self.mode == 'min' else -1

	def on_train_begin(self, model, **kwargs):

		if self.monitor in model.metrics:
			self.best_val = np.inf
			self.fails = 0

		else:
			print(f"Metrics:\n{model.metrics}")
			raise Exception(f"Metric argument in EarlyStopping constructor isn't exist: '{self.monitor}'")

	def on_epoch_begin(self, **kwargs):
		pass

	def on_batch_begin(self, **kwargs):
		pass
	
	def on_batch_end(self, **kwargs):
		pass
	
	def on_epoch_end(self, model, metrics, **kwargs):

		value = metrics[self.monitor].get_result() * self.sign

		print(f"Mode={self.mode}: {value} < {self.best_val} - {self.min_delta}")

		if value <= self.best_val - self.min_delta:
			self.best_val = value
			self.fails = 0

		else:
			self.fails += 1
			if self.fails > self.patience:
				model.stop_trainning = True

	def on_train_end(self, **kwargs):
		pass
