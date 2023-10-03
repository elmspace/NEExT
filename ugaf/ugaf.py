"""
	Author : Ash Dehghan
	Description:
"""

from ugaf.graph_collection import Graph_Collection

class UGAF:


	def __init__(self):
		self.graph_c = Graph_Collection()


	def build_graph_collection(self, config):
		"""
			This method take user config and build a graph collection.
		"""
		self.graph_c.load_graphs(config)
		