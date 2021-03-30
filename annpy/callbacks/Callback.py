from abc import ABCMeta, abstractmethod

class Callback():

	@abstractmethod
	def __init__(self):
		pass
	
	@abstractmethod
	def on_train_begin(self):
		pass
	
	@abstractmethod
	def on_train_end(self):
		pass
	
	@abstractmethod
	def on_epoch_begin(self):
		pass
	
	@abstractmethod
	def on_epoch_end(self):
		pass

	@abstractmethod
	def on_batch_begin(self):
		pass
	
	@abstractmethod
	def on_batch_end(self):
		pass

