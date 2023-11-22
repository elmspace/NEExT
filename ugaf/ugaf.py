"""
	Author : Ash Dehghan
"""

import umap
import scipy
import vectorizers
import numpy as np
import pandas as pd
from tqdm import tqdm
import plotly.express as px
from numpy import linalg as LA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# Internal Modules
from ugaf.ml_models import ML_Models
from ugaf.feature_engine import Feature_Engine
from ugaf.graph_collection import Graph_Collection
from ugaf.graph_embedding_engine import Graph_Embedding_Engine


class UGAF:


	def __init__(self):
		self.graph_c = Graph_Collection()
		self.feat_eng = Feature_Engine()
		self.ml_model = ML_Models()
		self.g_emb = Graph_Embedding_Engine()
		self.gc_status = False
		self.emb_cols = []
		self.sim_matrix_largets_eigen_values = []
		self.graph_embedding = {}
		self.graph_embedding_df = {}
		self.graph_emb_dim_reduced = {}


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


	def add_graph_labels(self, graph_label_csv_path):
		"""
			This function takes as input a csv file for graph labels and uses the pre-built
			graph collection object, to assign labels to graphs.
		"""
		self.graph_c.assign_graph_labels(graph_label_csv_path)
		

	def extract_graph_features(self, feature_config):
		"""
			This method will use the Feature Engine object to build features
			on the graph, which can then be used to compute graph embeddings
			and other statistics on the graph.
		"""
		for g_obj in tqdm(self.graph_c.graph_collection, desc="Building features"):
			G = g_obj["graph"]
			g_obj["graph_features"] = self.feat_eng.build_features(G, feature_config)


	def build_graph_embedding(self, graph_embedding_type):
		"""
		This method uses the Graph Embedding Engine object to 
		build a graph embedding for every graph in the graph collection.
		"""
		graphs_embed, graph_embedding_df = self.g_emb.build_graph_embedding(graph_embedding_type, graph_c = self.graph_c)
		

		emb_cols = ["emb_0", "emb_1", "emb_2", "emb_3", "emb_4"]
		reducer = umap.UMAP()
		redu_emb = reducer.fit_transform(graph_embedding_df[emb_cols])
		graph_embedding_df["x"] = redu_emb[:,0]
		graph_embedding_df["y"] = redu_emb[:,1]

		fig = px.scatter(graph_embedding_df, x="x", y="y", color="graph_id", size=[4]*len(graph_embedding_df))
		fig.show()





