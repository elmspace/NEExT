"""
	Author : Ash Dehghan
	Description:
"""

from ugaf.graph_collection import Graph_Collection

class UGAF:


	def __init__(self):
		self.graph_c = Graph_Collection()


	def build_graph_collection(self, edge_csv_path, node_graph_map_csv_path):
		"""
			This method uses the Graph Collection class to build an object
			which handels a set of graphs.
		"""
		self.graph_c.load_graphs(edge_csv_path, node_graph_map_csv_path)
		

	def filter_for_largest_cc(self):
		"""
			This method uses the Graph Collection class to filter the subgraphs
			to only contain the largest connected component.
		"""
		self.graph_c.filter_collection_for_largest_connected_component()