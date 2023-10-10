"""
	Author: Ash Dehghan
	Description: This class provides some builtin node and structural embedding
	solutions to be used by UGAF. The user could provide their own node embeddings.
"""

import random
import numpy as np
import networkx as nx
from loguru import logger
from node2vec import Node2Vec


class Embedding_Engine:


	def __init__(self):
		pass



	def run_embedding(self, G, embedding_type, emb_dim):
		"""
			This method runs a builtin embedding on nodes of graph G 
		"""
		if embedding_type == "node2vec":
			embeddings = self.run_node2vec_embedding(G, emb_dim)
		elif embedding_type == "deepwalk":
			embeddings = self.run_deepwalk_embedding(G, emb_dim)
		elif embedding_type == "lsme":
			embeddings = self.run_lsme_embedding(G, emb_dim)
		else:
			logger.error("Embedding type selected is not valid.")
		return embeddings



	def run_node2vec_embedding(self, G, emb_dim):
		"""
			This method takes as input a networkx graph object
			and runs a Node2Vec embedding using default values.
		"""
		node2vec = Node2Vec(G, dimensions=emb_dim, walk_length=5, num_walks=2, workers=2, quiet=True)
		model = node2vec.fit(window=2, min_count=1, batch_words=4)
		embeddings = {node: list(model.wv[node]) for node in G.nodes()}
		return embeddings


	def run_lsme_embedding(self, G, emb_dim):
		"""
			This method takes as input a networkx graph object
			and runs a LSME structural embedding.
		"""
		nodes = list(G.nodes)
		embeddings = {}
		for node in nodes:
			emb = self.lsme_run_random_walk(node, G, sample_size=50, rw_length=10)
			if len(emb) < emb_dim:
				emb += [0] * (emb_dim - len(emb))
			else:
				emb = emb[0:emb_dim]
			embeddings[node] = emb
		return embeddings


	def lsme_run_random_walk(self, root_node, G, sample_size=50, rw_length=30):
		"""
			This method runs the random-walk for the LSME algorithm
		"""
		walk = {}
		for i in range(sample_size):
			c_node = root_node
			for i in range(rw_length):
				neighbors = [n for n in G.neighbors(c_node)]
				n_node = random.choice(neighbors)
				dist_b = len(nx.shortest_path(G, root_node, c_node)) - 1
				dist_a = len(nx.shortest_path(G, root_node, n_node)) - 1
				c_node = n_node
				if dist_b in walk.keys():
					if dist_a in walk[dist_b].keys():
						walk[dist_b]["total"] += 1
						walk[dist_b][dist_a] += 1
					else:
						walk[dist_b]["total"] += 1
						walk[dist_b][dist_a] = 1
				else:
					walk[dist_b] = {}
					walk[dist_b]["total"] = 1
					walk[dist_b][dist_a] = 1
		max_walk = max(list(walk.keys()))
		emb = []
		for i in range(0, max_walk+1):
			step = walk[i]
			b = i-1
			s = i
			f = i+1
			pb = step[b]/step["total"] if b in step.keys() else 0
			ps = step[s]/step["total"] if s in step.keys() else 0
			pf = step[f]/step["total"] if f in step.keys() else 0
			emb += [pb, ps, pf]
		emb = emb[3:len(emb)+1]
		return emb



	def run_deepwalk_embedding(self, G, emb_dim):
		"""
			This method takes as input a networkx graph object
			and runs a DeepWalk node embedding.
		"""
		pass















