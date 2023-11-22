"""
	This class contains methods which allows the user
	to compute various features on the graph, which 
	could capture various properties of the graph
	including structural, density, ...
"""

# External Libraries
import umap
import json
import random
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
from sklearn.decomposition import PCA

# Internal Libraries
from ugaf.helper_functions import get_nodes_x_hops_away
from ugaf.node_embedding_engine import Node_Embedding_Engine


class Feature_Engine:


	def __init__(self):
		self.node_emb_engine = Node_Embedding_Engine()
		self.feature_functions = {}
		self.feature_functions["lsme"] = self.build_lsme
		self.feature_functions["basic_expansion"] = self.build_basic_expansion
		self.feature_functions["page_rank"] = self.build_page_rank
		self.feature_functions["degree_centrality"] = self.build_degree_centrality
		self.feature_functions["closeness_centrality"] = self.build_closeness_centrality


	def load_config(self, config_file_path):
		with open(config_file_path) as config_file:
			config = dict(json.load(config_file))
		return config


	def build_features(self, G, config_file_path):
		config = self.load_config(config_file_path)
		node_samples = self.sample_graph(G, config)
		feature_collection = {}
		feature_collection["graph_features"] = {}
		for func_name in config["features"]:
			feature_collection = self.feature_functions[func_name](feature_collection, G, config["features"][func_name], func_name, node_samples)
		feature_collection = self.build_gloabal_embedding(feature_collection, node_samples, config)
		return feature_collection


	def sample_graph(self, G, config):
		"""
			This method will sample the graph based on the information
			given in the configuration.
		"""
		if config["graph_sample"]["flag"] == "no":
			node_samples = list(G.nodes)
		else:
			sample_size = int(len(G.nodes) * config["graph_sample"]["sample_fraction"])
			graph_nodes = list(G.nodes)[:]
			random.shuffle(graph_nodes)
			node_samples = graph_nodes[0:sample_size]
		return node_samples


	def build_gloabal_embedding(self, feature_collection, node_samples, config):
		"""
			This method will use the features built on the graph to construct
			a global embedding for the nodes of the graph.
		"""
		feature_collection["global_embedding"] = {}
		if config["gloabl_embedding"]["type"] == "concat":
			for node in node_samples:
				feature_collection["global_embedding"][node] = []
				for func_name in config["features"]:
					if "embs" in feature_collection["graph_features"][func_name]:
						feature_collection["global_embedding"][node] += feature_collection["graph_features"][func_name]["embs"][node]
		else:
			raise ValueError("Gloabl embedding type is not supported.")
		# Reduce the dimension of the gloabal embedding (if flag is yes)
		if config["gloabl_embedding"]["compression"]["flag"] == "yes":
			emb_keys = list(feature_collection["global_embedding"].keys())
			embs = np.array([value for value in feature_collection["global_embedding"].values()])
			reducer = PCA(n_components=config["gloabl_embedding"]["compression"]["size"])
			embs_reduced = reducer.fit_transform(embs)
			for idx, key_val in enumerate(emb_keys):
				feature_collection["global_embedding"][key_val] = list(embs_reduced[idx])
		return feature_collection


	def build_page_rank(self, feature_collection, G, config, func_name, node_samples):
		"""
			This method will compute page rank for every node up to 
			emb_dim hops away neighbors.
		"""
		emb_dim = int(config["emb_dim"])
		pr = nx.pagerank(G, alpha=0.9)
		embs = {}
		for node in list(G.nodes):
			embs[node] = []
			nbs = get_nodes_x_hops_away(G, node, max_hop_length=emb_dim)
			embs[node].append(pr[node])
			for i in range(1, emb_dim+1):
				if i in nbs:
					nbs_pr = [pr[j] for j in nbs[i]]
					embs[node].append(sum(nbs_pr)/len(nbs_pr))
				else:
					embs[node].append(0.0)
		feature_collection["graph_features"][func_name] = {}
		feature_collection["graph_features"][func_name]["embs"] = embs
		return feature_collection


	def build_degree_centrality(self, feature_collection, G, config, func_name, node_samples):
		"""
			This method will compute degree centrality for every node up to 
			emb_dim hops away neighbors.
		"""
		emb_dim = int(config["emb_dim"])
		pr = nx.degree_centrality(G)
		embs = {}
		for node in list(G.nodes):
			embs[node] = []
			nbs = get_nodes_x_hops_away(G, node, max_hop_length=emb_dim)
			embs[node].append(pr[node])
			for i in range(1, emb_dim+1):
				if i in nbs:
					nbs_pr = [pr[j] for j in nbs[i]]
					embs[node].append(sum(nbs_pr)/len(nbs_pr))
				else:
					embs[node].append(0.0)
		feature_collection["graph_features"][func_name] = {}
		feature_collection["graph_features"][func_name]["embs"] = embs
		return feature_collection


	def build_closeness_centrality(self, feature_collection, G, config, func_name, node_samples):
		"""
			This method will compute closeness centrality for every node up to 
			emb_dim hops away neighbors.
		"""
		emb_dim = int(config["emb_dim"])
		pr = nx.closeness_centrality(G)
		embs = {}
		for node in list(G.nodes):
			embs[node] = []
			nbs = get_nodes_x_hops_away(G, node, max_hop_length=emb_dim)
			embs[node].append(pr[node])
			for i in range(1, emb_dim+1):
				if i in nbs:
					nbs_pr = [pr[j] for j in nbs[i]]
					embs[node].append(sum(nbs_pr)/len(nbs_pr))
				else:
					embs[node].append(0.0)
		feature_collection["graph_features"][func_name] = {}
		feature_collection["graph_features"][func_name]["embs"] = embs
		return feature_collection


	def build_lsme(self, feature_collection, G, config, func_name, node_samples):
		emb_dim = config["emb_dim"]
		embs = self.node_emb_engine.run_lsme_embedding(G, emb_dim, node_samples)
		feature_collection["graph_features"][func_name] = {}
		feature_collection["graph_features"][func_name]["embs"] = embs
		return feature_collection


	def build_basic_expansion(self, feature_collection, G, config, func_name, node_samples):
		emb_dim = config["emb_dim"]
		embs = self.node_emb_engine.run_expansion_embedding(G, emb_dim, node_samples)
		feature_collection["graph_features"][func_name] = {}
		feature_collection["graph_features"][func_name]["embs"] = embs
		return feature_collection

		

		

