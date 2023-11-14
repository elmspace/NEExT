"""
	Author : Ash Dehghan
"""

import umap
import scipy
import vectorizers
import numpy as np
import pandas as pd
from tqdm import tqdm
from numpy import linalg as LA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from ugaf.ml_models import ML_Models
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from ugaf.graph_collection import Graph_Collection
from ugaf.feature_engine import Feature_Engine
from sklearn.metrics.pairwise import cosine_similarity


class UGAF:


	def __init__(self):
		self.graph_c = Graph_Collection()
		self.feat_eng = Feature_Engine()
		self.ml_model = ML_Models()
		self.gc_status = False
		self.emb_cols = []
		self.sim_matrix_largets_eigen_values = []
		self.graph_embedding = {}
		self.graph_embedding_df = {}
		self.graph_emb_dim_reduced = {}


	def check_gc_status(func):
		def _check_gc_status(self, *args, **kwargs):
			if not self.gc_status:
				raise ValueError("You need to build a graph collection first.")
				exit(0)
			func(self, *args, **kwargs)
		return _check_gc_status


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
	def add_graph_labels(self, graph_label_csv_path):
		"""
			This function takes as input a csv file for graph labels and uses the pre-built
			graph collection object, to assign labels to graphs.
		"""
		self.graph_c.assign_graph_labels(graph_label_csv_path)
		

	@check_gc_status
	def extract_graph_features(self, feature_config):
		"""
			This method will use the Feature Engine object to build features
			on the graph, which can then be used to compute graph embeddings
			and other statistics on the graph.
		"""
		for g_obj in tqdm(self.graph_c.graph_collection, desc="Building features"):
			G = g_obj["graph"]
			g_obj["graph_features"] = self.feat_eng.build_features(G, feature_config)

		



