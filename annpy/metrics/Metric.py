import copy
from abc import ABCMeta, abstractmethod

# Kwargs is necessary in @abstractmethod because some childs metrics would not take arguments of other metrics
class Metric(metaclass=ABCMeta):

	def __init__(self):
		self.count = 0
		self.total = 0
		self.mem = []
		self.name = None

	def __str__(self):
		return self.name

	def set_name(self, name):
		self.name = name
		return self

	def log(self):
		return f"-- {str(self)}: {self.get_result()} "

	def __call__(self, predictions, targets):
		
		# total = self.get_mem_len_append(predictions, targets)
		self.count += self.compute(predictions, targets)
		self.total += 1
		# self.total += total

	def save_result(self):
		self.mem.append(self.get_result())

	@abstractmethod
	def compute(self):
		pass

	# @abstractmethod
	# def get_mem_len_append(self, **kwargs):
	# 	pass

	def get_result(self):
		# if self.total == 0:
		# 	return self.mem[-1]
		return self.count / self.total

	def get_mem(self):
		return self.mem

	def reset(self, save):
		if save:
			self.save_result()
		self.count = 0
		self.total = 0

	def hard_reset(self):
		self.reset(save=False)
		self.mem = []

	@abstractmethod
	def summary(self, **kwargs):
		pass

	def copy(self):
		return copy.deepcopy(self)

	# def append_into(self, metrics, val_duplication=True):

	# 	metrics[str(self)] = self

	# 	if val_duplication:
	# 		cpy = self.copy().set_name('val_' + str(self))
	# 		metrics[str(cpy)] = cpy
