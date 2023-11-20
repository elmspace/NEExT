"""
	This class contains methods which allows the user
	to compute various features on the graph, which 
	could capture various properties of the graph
	including structural, density, ...
"""

import umap
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import PCA
from ugaf.node_embedding_engine import Node_Embedding_Engine


class Feature_Engine:


	def __init__(self):
		self.node_emb_engine = Node_Embedding_Engine()
		self.feature_functions = {}
		self.feature_functions["lsme"] = self.build_lsme
		self.feature_functions["basic_expansion"] = self.build_basic_expansion


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

		

		

