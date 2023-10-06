"""
	Author: Ash Dehghan
	Description: This class provides some builtin node and structural embedding
	solutions to be used by UGAF. The user could provide their own node embeddings.
"""

import numpy as np
from loguru import logger
from node2vec import Node2Vec


class Embedding_Engine:


	def __init__(self):
		pass



	def run_embedding(self, G, embedding_type):
		"""
			This method runs a builtin embedding on nodes of graph G 
		"""
		if embedding_type == "node2vec":
			embeddings = self.run_node2vec_embedding(G)
		else:
			logger.error("Embedding type selected is not valid.")
		return embeddings



	def run_node2vec_embedding(self, G):
		"""
			This method takes as input a networkx graph object
			and runs a Node2Vec embedding using default values.
		"""
		node2vec = Node2Vec(G, dimensions=4, walk_length=5, num_walks=2, workers=2, quiet=True)
		model = node2vec.fit(window=2, min_count=1, batch_words=4)
		embeddings = {node: list(model.wv[node]) for node in G.nodes()}
		return embeddings