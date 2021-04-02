from abc import ABCMeta, abstractmethod

# Kwargs is necessary because some childs callbacks would not take arguments of other callbacks
class Callback():

	@abstractmethod
	def __init__(self):
		pass

	@abstractmethod
	def on_train_begin(self, **kwargs):
		pass
	
	@abstractmethod
	def on_train_end(self, **kwargs):
		pass
	
	@abstractmethod
	def on_epoch_begin(self, **kwargs):
		pass
	
	@abstractmethod
	def on_epoch_end(self, **kwargs):
		pass

	@abstractmethod
	def on_batch_begin(self, **kwargs):
		pass

	@abstractmethod
	def on_batch_end(self, **kwargs):
		pass

	@abstractmethod
	def summary(self):
		pass
