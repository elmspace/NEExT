"""
	Author : Ash Dehghan
	Description:
"""

import scipy
import vectorizers
import numpy as np
import pandas as pd
from tqdm import tqdm
from loguru import logger
from ugaf.graph_collection import Graph_Collection
from ugaf.embedding_engine import Embedding_Engine


class UGAF:


	def __init__(self):
		self.graph_c = Graph_Collection()
		self.emb_eng = Embedding_Engine()
		self.gc_status = False


	def check_gc_status(func):
		def _check_gc_status(self):
			if not self.gc_status:
				logger.error("You need to build a graph collection first.")
				exit(0)
		return _check_gc_status


	@check_gc_status
	def print_gc_info(self):
		"""
			This method will print out some simple information about the graph collection.
		"""
		


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
		self.gc_status = True
		

	@check_gc_status
	def build_node_embedding(self, embedding_type, emb_dim):
		# Run embedding calculation
		logger.info("Running %s embedding" % (embedding_type))
		for g_obj in tqdm(self.graph_c.graph_collection, desc="Building embeddings"):
			G = g_obj["graph"]
			embeddings = self.emb_eng.run_embedding(G, embedding_type, emb_dim)
			g_obj["embedding"] = {}
			g_obj["embedding"][embedding_type] = embeddings


	@check_gc_status
	def build_graph_embedding(self, using_embedding):
		logger.info("Creating incidence matrix")
		n = self.graph_c.total_numb_of_nodes
		rows = self.graph_c.graph_id_node_array
		cols = np.arange(n)
		incidence_matrix = scipy.sparse.csr_matrix((np.repeat(1.0,n).astype(np.float32), (rows, cols)))

		embedding_collection = []
		for g_obj in tqdm(self.graph_c.graph_collection, desc="Loading embeddings"):
			if "embedding" not in g_obj:
				logger.error("You have to run node/structural embedding first.")
			if using_embedding not in g_obj["embedding"]:
				logger.error("No such selected embedding.")
			embs = g_obj["embedding"][using_embedding]
			embedding_collection.append(list(embs.values()))

		embedding_collection = np.array(embedding_collection, dtype=object)
		embedding_collection = np.vstack(embedding_collection)

		graphs_embed = vectorizers.ApproximateWassersteinVectorizer(
			normalization_power=0.66,
			random_state=42,
		).fit_transform(incidence_matrix, vectors=embedding_collection)
		
		