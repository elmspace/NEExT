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
from karateclub import DeepWalk


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
		elif embedding_type == "expansion":
			embeddings = self.run_expansion_embedding(G, emb_dim)
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
		model =  DeepWalk(dimensions=emb_dim)
		model.fit(G)
		dw_emb = model.get_embedding()
		embeddings = {}
		for index, node in enumerate(list(G.nodes)):
			embeddings[node] = list(dw_emb[index])
		return embeddings


	def run_expansion_embedding(self, G, emb_dim):
		"""
			This method takes as input a networkx graph object
			and runs a simple expansion property embedding.
		"""
		embeddings = {}
		d = (2 * len(G.edges))/len(G.nodes)
		for node in G.nodes:
			dist_list = self.get_numb_of_nb_x_hops_away(G, node, emb_dim)
			norm_list = []
			for i in range(len(dist_list)):
				if i == 0:
					norm_val = 1 * d
				else:
					norm_val = dist_list[i] - 1
					if norm_val <= 0:
						norm_val = 1
				norm_list.append(norm_val * d)
			emb = [dist_list[i]/norm_list[i] for i in range(len(dist_list))]
			embeddings[node] = emb
		return embeddings
					

	def get_numb_of_nb_x_hops_away(self, G, node, max_hop_length):
		"""
			This method will compute the number of neighbors x hops away from
			a given node.
		"""
		node_dict = {}
		dist_dict = {}
		node_list = [node]
		keep_going = True
		while keep_going:
			n = node_list.pop(0)
			nbs = G.neighbors(n)
			for nb in nbs:
				if (nb not in node_dict) and (nb != node):
					node_list.append(nb)
					dist_to_source = len(nx.shortest_path(G, source=node, target=nb)) - 1
					node_dict[nb] = dist_to_source
					if dist_to_source not in dist_dict:
						dist_dict[dist_to_source] = [nb]
					else:
						dist_dict[dist_to_source].append(nb)
					if dist_to_source >= max_hop_length:
						keep_going = False
			if len(node_list) == 0:
				keep_going = False
		# Build dist list
		max_hop = max(list(dist_dict.keys()))
		hop_list = []
		for i in range(1, max_hop+1):
			hop_list.append(len(dist_dict[i]))
		hop_list = hop_list + [0]*(max_hop_length-len(hop_list))
		return hop_list














