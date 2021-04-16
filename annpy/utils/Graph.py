# import matplotlib.pyplot as plt
# import numpy as np

class Graph():
	pass

# 	def __init__(self, graphs=None):
		
# 		if graphs:
# 			self.graphs = graphs
# 		else:
# 			self.graphs = {}

# 	def add_graph(self, name='Default name', x_axis='epochs', y_axis='y'):
		
# 		# if len(axes) != 3:
# 		# 	raise Exception("Parameter 'axes' len need to be 3")

# 		graphs = {
# 			'graph 1': {
# 				'x' = [],
# 				'y' = [[]],
# 			},
# 			'graph 2': {
# 				'x' = [],
# 				'y' = [[], []],
# 			}
# 		}
# 		# plt.setp(axs[:], xlabel='x axis label')
# 		# self.graphs.append(graph)
# 		self.graphs[name] = {'x': None, 'y': []}
# 		# self.graphs[name] = {x_axis: None, y_axis: [None]}

# 	def add_curve(self, curve, graph='Default name'):

# 		if graph not in self.graphs:
# 			raise Exception(f"Unable to find {graph} graph")

# 		if self.graphs[graph]['x']:
# 			if range(len(curve)) != self.graphs[graph]['x']:
# 				raise Exception(f"x axis of graph {graph} are not the same: {range(len(curve))} != {self.graphs[graph]['x']}")
# 		else:
# 			# self.graphs[graph]['x'] = range(len(curve))
# 			self.graphs[graph]['x'] = np.linspace(0, range(len(curve)), 1)

# 		self.graphs[graph]['y'].append(curve)

# 	def show(self, graphs=[]):

# 		print(f"{self}:\n{self.graphs}")
		
# 		fig, graphs_lst = plt.subplots(ncols=2)
# 		for graph in graphs_lst:
# 			graph.
# 		plt.show()

# """
# x=
# plt.plot(x,np.sin(x))

# fig, graph_lst = plt.subplots(ncols=2)
# """