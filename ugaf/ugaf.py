"""
	Author : Ash Dehghan
	Description:
"""

from tqdm import tqdm
from loguru import logger
from ugaf.graph_collection import Graph_Collection
from ugaf.embedding_engine import Embedding_Engine


class UGAF:



	def __init__(self):
		self.graph_c = Graph_Collection()
		self.emb_eng = Embedding_Engine()



	def build_graph_collection(self, edge_csv_path, node_graph_map_csv_path, filter_for_largest_cc=True, reset_node_indices=True):
		"""
			This method uses the Graph Collection class to build an object
			which handels a set of graphs.
		"""
		self.graph_c.load_graphs(edge_csv_path, node_graph_map_csv_path)
		if filter_for_largest_cc:
			self.graph_c.filter_collection_for_largest_connected_component()
		if reset_node_indices:
			self.graph_c.reset_node_indices()
		


	def build_node_embedding(self, embedding_type):
		# Check if graph collection object was built
		if not self.graph_c.gc_status:
			logger.error("You need to build a graph collection first.")
		# Run embedding calculation
		logger.info("Running %s embedding" % (embedding_type))
		for g_obj in tqdm(self.graph_c.graph_collection, desc="Building embeddings"):
			G = g_obj["graph"]
			embeddings = self.emb_eng.run_embedding(G, embedding_type)
			g_obj["embedding"] = {}
			g_obj["embedding"][embedding_type] = embeddings


	def build_graph_embedding(self):
		pass