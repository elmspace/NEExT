"""
	Author : Ash Dehghan
"""

import umap
import scipy
import vectorizers
import numpy as np
import pandas as pd
from tqdm import tqdm
from loguru import logger
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from ugaf.graph_collection import Graph_Collection
from ugaf.embedding_engine import Embedding_Engine


class UGAF:


	def __init__(self):
		self.graph_c = Graph_Collection()
		self.emb_eng = Embedding_Engine()
		self.gc_status = False
		self.graph_embedding = {}
		self.graph_emb_dim_reduced = {}

	def check_gc_status(func):
		def _check_gc_status(self, *args, **kwargs):
			if not self.gc_status:
				logger.error("You need to build a graph collection first.")
				exit(0)
			func(self, *args, **kwargs)
		return _check_gc_status


	@check_gc_status
	def get_gc_stats(self):
		"""
			This method will print out some simple information about the graph collection.
		"""
		graph_c_stats = self.graph_c.export_graph_collection_stats()

		# Plot the node distribution
		x = graph_c_stats["numb_node_dist"]
		plt.figure()
		plt.hist(x)
		plt.xlabel('Numb of Nodes', fontsize=12)
		plt.ylabel('Freq', fontsize=12)
		plt.title("Distribution of Number of Nodes per Graph", fontsize=14)

		# Plot avg node degree distribution
		x = graph_c_stats["avg_node_degree"]
		plt.figure()
		plt.hist(x)
		plt.xlabel('Avg Node Degree', fontsize=12)
		plt.ylabel('Freq', fontsize=12)
		plt.title("Distribution of Average Node Degree per Graph", fontsize=14)

		plt.show()


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
		"""
			This function takes as input the embedding type, and dimension
			and uses the embedding object to build embeddings for the nodes of 
			graphs in the graph collection.
		"""
		logger.info("Running %s embedding" % (embedding_type))
		for g_obj in tqdm(self.graph_c.graph_collection, desc="Building embeddings"):
			G = g_obj["graph"]
			embeddings = self.emb_eng.run_embedding(G, embedding_type, emb_dim)
			g_obj["embedding"] = {}
			g_obj["embedding"][embedding_type] = embeddings
		self.normalize_embedding(embedding_type)


	@check_gc_status
	def normalize_embedding(self, embedding_type):
		"""
			This method normalizes the embedding vectors using
			Scikit-Learn standard scaler.
		"""
		logger.info("Normalizing the embedding data")
		for g_obj in tqdm(self.graph_c.graph_collection, desc="Normalizing"):
			embs = g_obj["embedding"][embedding_type]
			scaler = StandardScaler()
			embs_df = pd.DataFrame(embs)
			embs_df_cols = embs_df.columns.tolist()
			embs_df = pd.DataFrame(scaler.fit_transform(embs_df))
			embs_df.columns = embs_df_cols
			g_obj["embedding"][embedding_type] = embs_df.to_dict(orient='list')


	@check_gc_status
	def build_graph_embedding(self, source_embedding):
		logger.info("Creating incidence matrix")
		n = self.graph_c.total_numb_of_nodes
		rows = self.graph_c.graph_id_node_array
		cols = np.arange(n)
		incidence_matrix = scipy.sparse.csr_matrix((np.repeat(1.0,n).astype(np.float32), (rows, cols)))

		embedding_collection = []
		for g_obj in tqdm(self.graph_c.graph_collection, desc="Loading embeddings"):
			if "embedding" not in g_obj:
				logger.error("You have to run node/structural embedding first.")
			if source_embedding not in g_obj["embedding"]:
				logger.error("No such selected embedding.")
			embs = g_obj["embedding"][source_embedding]
			embedding_collection.append(list(embs.values()))

		embedding_collection = np.array(embedding_collection, dtype=object)
		embedding_collection = np.vstack(embedding_collection)

		graphs_embed = vectorizers.ApproximateWassersteinVectorizer(
			normalization_power=0.66,
			random_state=42,
		).fit_transform(incidence_matrix, vectors=embedding_collection)
		self.graph_embedding[source_embedding] = graphs_embed
		

	@check_gc_status
	def reduce_graph_embedding_dimension(self, algorithm, source_embedding, plot=False):
		"""
			This function uses the graph embeddings built using source node/structural embedding
			and saves it, and also plots it, if <plot> flag is set to True.
		"""
		# Check if source embedding exists
		if not source_embedding in self.graph_embedding:
			logger.error("Missing graph embedding for source: %s" %(source_embedding))
			exit(0)
		if algorithm == "umap":
			logger.info("Running UMAP")
			reducer = umap.UMAP()
			redu_emb = reducer.fit_transform(self.graph_embedding[source_embedding])
			self.graph_emb_dim_reduced["umap"] = {}
			self.graph_emb_dim_reduced["umap"]["x"] = redu_emb[:,0]
			self.graph_emb_dim_reduced["umap"]["y"] = redu_emb[:,1]
		elif algorithm == "tsne":
			logger.info("Running tSNE")
			reducer = TSNE(n_components=2, random_state=0)
			redu_emb = reducer.fit_transform(self.graph_embedding[source_embedding])
			self.graph_emb_dim_reduced["tsne"] = {}
			self.graph_emb_dim_reduced["tsne"]["x"] = redu_emb[:,0]
			self.graph_emb_dim_reduced["tsne"]["y"] = redu_emb[:,1]
		elif algorithm == "pca":
			logger.info("Running PCA")
			reducer = PCA(n_components=2)
			redu_emb = reducer.fit_transform(self.graph_embedding[source_embedding])
			self.graph_emb_dim_reduced["pca"] = {}
			self.graph_emb_dim_reduced["pca"]["x"] = redu_emb[:,0]
			self.graph_emb_dim_reduced["pca"]["y"] = redu_emb[:,1]
		else:
			logger.error("Selected algorithm is not supported.")

		if plot:
			x = self.graph_emb_dim_reduced[algorithm]["x"]
			y = self.graph_emb_dim_reduced[algorithm]["y"]
			plt.figure()
			plt.scatter(x, y)
			plt.xlabel('Dim-1', fontsize=12)
			plt.ylabel('Dim-2', fontsize=12)
			plt.title(algorithm.upper(), fontsize=14)
			plt.show()