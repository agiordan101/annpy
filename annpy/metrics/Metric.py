from abc import ABCMeta, abstractmethod

import numpy as np
import plotly.express as px
# import plotly.graph_objects as go

def metrics_data_to_str(metrics_lst):

	metrics_log = ""
	for metric in metrics_lst:
		# print(metric.get_obj_name())
		# print(metric.count)
		# print(metric.total)
		metrics_log += str(metric)
		metric.reset()
	return metrics_log

class Metric(metaclass=ABCMeta):

	@abstractmethod
	def __init__(self):
		self.count = 0
		self.total = 0
		self.mem = []

	@abstractmethod
	def compute(self, prediction, target):
		pass

	@abstractmethod
	def get_mem_len_append(self, predictions, targets):
		pass

	def __call__(self, predictions, targets, update_mem=True):
		count = self.compute(predictions, targets)
		total = self.get_mem_len_append(predictions, targets)
		self.count += count
		self.total += total
		if update_mem:
			self.mem.append(count / total)

	def __str__(self):
		result = self.get_result()
		if not isinstance(result, float):
			result = np.mean(result)
		# print(f"result {result}")
		return f" -- {self.get_obj_name()}: {result}"

	@abstractmethod
	def get_obj_name(self):
		pass

	def get_result(self):
		return self.count / self.total

	def get_mem(self):
		return self.mem

	def reset(self):
		self.count = 0
		self.total = 0

	def hard_reset(self):
		self.reset()
		self.mem = []

	@abstractmethod
	def summary(self):
		pass

	# def print_graph(self):
	# 	print(f"{self.get_obj_name()} {len(self.mem)}: {self.mem}")
	# 	# fig = px.line(self.mem, x=list(range(len(self.mem))), y=self.mem)
	# 	# fig = px.line(self.mem, x="epochs", y=self.get_obj_name())
	# 	fig = px.line(self.mem)
	# 	fig.show()
