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

	def on_train_begin(self, **kwargs):

		self.metric = None
		print(f"Metrics:\n{self.model.current_metrics}")
		for m in self.model.current_metrics:
			
			if str(m) == self.monitor:
				self.metric = m
				break

		if not self.metric:
			print(f"Metrics:\n{self.model.current_metrics}")
			raise Exception(f"EarlyStopping constructor: Unable to find monitored metric {self.monitor} in this model")

		self.best_val = np.inf
		self.fails = 0

		if self.mode == 'auto':
			self.mode = self.metric.get_variation_goal()
		
		self.sign = 1 if self.mode == 'min' else -1

	def on_epoch_begin(self, **kwargs):
		pass

	def on_batch_begin(self, **kwargs):
		pass
	
	def on_batch_end(self, **kwargs):
		pass
	
	def on_epoch_end(self, verbose=True, **kwargs):

		value = self.metric.get_result() * self.sign
		# value = self.model.metrics[self.monitor].get_result() * self.sign

		# print(f"Mode={self.mode}: {value} < {self.best_val} - {self.min_delta}")

		if value <= self.best_val - self.min_delta:
			self.best_val = value
			self.fails = 0

		else:
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
