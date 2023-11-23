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
		self.emb_cols = []
		self.graph_embedding = {}


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
			graph_id = g_obj["graph_id"]
			g_obj["graph_features"] = self.feat_eng.build_features(G, graph_id, feature_config)
		self.standardize_graph_features_globaly()


	def standardize_graph_features_globaly(self):
		"""
			This method will standardize the graph features across all graphs.
		"""
		all_graph_feats = pd.DataFrame()
		for g_obj in tqdm(self.graph_c.graph_collection, desc="Building features"):
			df = g_obj["graph_features"]["global_embedding"]
			if all_graph_feats.empty:
				all_graph_feats = df.copy(deep=True)
			else:
				all_graph_feats = pd.concat([all_graph_feats, df])

		# Grab graph feature embedding columns
		emb_cols = []
		for col in all_graph_feats.columns.tolist():
			if "emb_" in col:
				emb_cols.append(col)
		# Standardize the embedding
		node_ids = all_graph_feats["node_id"].tolist()
		graph_ids = all_graph_feats["graph_id"].tolist()
		emb_df = all_graph_feats[emb_cols].copy(deep=True)
		# Normalize data
		scaler = StandardScaler()
		emb_df = pd.DataFrame(scaler.fit_transform(emb_df))
		emb_df.columns = emb_cols
		emb_df.insert(0, "node_id", node_ids)
		emb_df.insert(1, "graph_id", graph_ids)
		# Keep a collective global embedding
		self.graph_c.global_embeddings = emb_df.copy(deep=True)
		self.graph_c.global_embeddings_cols = emb_cols
		# Re-assign global embeddings to each graph
		for g_obj in tqdm(self.graph_c.graph_collection, desc="Updating features"):
			df = emb_df[emb_df["graph_id"] == g_obj["graph_id"]].copy(deep=True)
			g_obj["graph_features"]["global_embedding"] = df


	def compute_similarity_matrix_stats(self, use_labels=False):
		"""
			This method will run through the features computes on the graph and computes
			similarity matrices on those features per graph.
		"""
		eigen_val_df = pd.DataFrame()
		for g_obj in tqdm(self.graph_c.graph_collection, desc="Computing similarity stats"):
			feature_list = []
			eigen_val_list = []
			graph_label = g_obj["graph_id"]
			if use_labels:
				graph_label = self.graph_c.grpah_labels_df[self.graph_c.grpah_labels_df["graph_id"] == g_obj["graph_id"]].iloc[0]["graph_label"]

			for feature in g_obj["graph_features"]["features"]:
				
				data = self.graph_c.global_embeddings[self.graph_c.global_embeddings["graph_id"] == g_obj["graph_id"]]
				embs_cols = g_obj["graph_features"]["features"][feature]["embs_cols"]
				
				data = data[embs_cols].values
			
				sim_matrix = cosine_similarity(data, data)

				eigenvalues, eigenvectors = LA.eig(sim_matrix)
				eigenvalues = [i.real for i in eigenvalues]

				max_ei = max(eigenvalues)

				feature_list.append(feature)
				eigen_val_list.append(max_ei)

			df = pd.DataFrame()
			df["feature"] = feature_list
			df["eigen_val"] = eigen_val_list
			df.insert(0, "graph_label", graph_label)

			if eigen_val_df.empty:
				eigen_val_df = df.copy(deep=True)
			else:
				eigen_val_df = pd.concat([eigen_val_df, df])

		fig = px.scatter(eigen_val_df, x="graph_label", y="eigen_val", color="feature")
		fig.show()

	def build_graph_embedding(self, graph_embedding_type):
		"""
			This method uses the Graph Embedding Engine object to 
			build a graph embedding for every graph in the graph collection.
		"""
		graph_embedding, graph_embedding_df = self.g_emb.build_graph_embedding(graph_embedding_type, graph_c = self.graph_c)
		self.graph_embedding = {}
		self.graph_embedding["graph_embedding"] = graph_embedding
		self.graph_embedding["graph_embedding_df"] = graph_embedding_df


	def visualize_graph_embedding(self, color_by_label=False):
		"""
			This method uses the the graph embedding and UMAP to
			visulize the embeddings in two dimensions. It can also color the
			points if there are labels available for the graph.
		"""
		if color_by_label:
			data = self.graph_embedding["graph_embedding_df"].merge(self.graph_c.grpah_labels_df, on="graph_id", how="inner")
		else:
			data = self.graph_embedding["graph_embedding_df"].copy(deep=True)
		# Identify embedding colomns
		emb_cols = []
		for col in data.columns.tolist():
			if "emb" in col:
				emb_cols.append(col)
		# Perform dimensionality reduction
		reducer = umap.UMAP()
		redu_emb = reducer.fit_transform(data[emb_cols])
		data["x"] = redu_emb[:,0]
		data["y"] = redu_emb[:,1]
		# Generate plotly figures
		if color_by_label:
			fig = px.scatter(data, x="x", y="y", color="graph_label", size=[4]*len(data))
		else:
			fig = px.scatter(data, x="x", y="y", size=[4]*len(data))
		# Update figure layout
		fig.update_layout(paper_bgcolor='white')
		fig.update_layout(plot_bgcolor='white')
		fig.update_yaxes(color='black')
		fig.update_layout(
			yaxis = dict(
				title = "Dim-1",
				zeroline=True,
				showline = True,
				linecolor = 'black',
				mirror=True,
				linewidth = 2
			),
			xaxis = dict(
				title = 'Dim-2',
				mirror=True,
				zeroline=True,
				showline = True,
				linecolor = 'black',
				linewidth = 2,
				tickangle = 90,
			),
			width=500,
			height=500,
			font=dict(
			size=15,
			color="black")
				
		)
		fig.update_layout(showlegend=True)
		fig.update_layout(legend=dict(
			yanchor="bottom",
			y=0.01,
			xanchor="left",
			x=0.78,
			bordercolor="Black",
			borderwidth=1
		))
		fig.update_xaxes(showgrid=False, gridwidth=0.5, gridcolor='#e3e1e1')
		fig.update_yaxes(showgrid=False, gridwidth=0.5, gridcolor='grey')
		fig.update_traces(marker_line_color='black', marker_line_width=1.5, opacity=0.6)
		fig.show()





