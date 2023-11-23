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
			g_obj["graph_features"] = self.feat_eng.build_features(G, feature_config)


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





