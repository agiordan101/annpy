from annpy.callbacks.Callback import Callback

class EarlyStopping(Callback):

	def __init__(self,
					monitor='val_loss',
					min_delta=0,
					patience=0,
					mode='auto'):
		# self.monitor_name = monitor
		pass
