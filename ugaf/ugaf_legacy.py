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
from ugaf.embedding_engine import Embedding_Engine
from ugaf.feature_engine import Feature_Engine
from sklearn.metrics.pairwise import cosine_similarity


class UGAF:


	def __init__(self):
		self.graph_c = Graph_Collection()
		self.emb_eng = Embedding_Engine()
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
		









	def extract_graph_features(self, feature_config):
		"""
			This method will use the Feature Engine object to build features
			on the graph, which can then be used to compute graph embeddings
			and other statistics on the graph.
		"""
		for g_obj in tqdm(self.graph_c.graph_collection, desc="Building features"):
			G = g_obj["graph"]

			feat_obj = self.feat_eng.build_features(G, feature_config)

			exit(0)
		











	@check_gc_status
	def build_node_embedding(self, embedding_type, emb_dim):
		"""
			This function takes as input the embedding type, and dimension
			and uses the embedding object to build embeddings for the nodes of 
			graphs in the graph collection.
		"""
		for g_obj in tqdm(self.graph_c.graph_collection, desc="Building embeddings"):
			G = g_obj["graph"]
			embeddings = self.emb_eng.run_embedding(G, embedding_type, emb_dim)
			g_obj["embedding"][embedding_type] = embeddings
			self.graph_c.built_embeddings.add(embedding_type)
		self.normalize_embedding(embedding_type)


	def build_custom_node_embedding(self, emb_func, emb_func_name, **kwargs):
		"""
			This method allows the user to create their own embedding algorithm
			and pass it into the ugaf object. The user can pass whatever arguments
			the users needs for their embedding algorithm.
		"""
		for g_obj in tqdm(self.graph_c.graph_collection, desc="Building embeddings"):
			G = g_obj["graph"]
			embeddings = emb_func(G, **kwargs)
			g_obj["embedding"][emb_func_name] = embeddings
			self.graph_c.built_embeddings.add(emb_func_name)
		self.normalize_embedding(emb_func_name)


	@check_gc_status
	def normalize_embedding(self, embedding_type):
		"""
			This method normalizes the embedding vectors using
			Scikit-Learn standard scaler.
		"""
		for g_obj in tqdm(self.graph_c.graph_collection, desc="Normalizing"):
			embs = g_obj["embedding"][embedding_type]
			scaler = StandardScaler()
			embs_df = pd.DataFrame(embs)
			embs_df_cols = embs_df.columns.tolist()
			embs_df = pd.DataFrame(scaler.fit_transform(embs_df))
			embs_df.columns = embs_df_cols
			g_obj["embedding"][embedding_type] = embs_df.to_dict(orient='list')


	@check_gc_status
	def build_graph_embedding(self, source_node_embedding, graph_embedding_type):
		if graph_embedding_type == "wasserstein":
			self.build_wasserstein_graph_embedding(source_node_embedding)
		elif graph_embedding_type == "similarity_matrix_eigen_vector":
			self.build_sim_matrix_eigen_vector_graph_embedding(source_node_embedding)
		else:
			raise ValueError("Graph embedding %s not supported." %(graph_embedding_type))


	def build_sim_matrix_eigen_vector_graph_embedding(self, source_node_embedding):
		"""
			This method method uses the input source node embedding and calculates
			a similarity matrix for embeddings of the same graph, and then solves for
			the eigen vector assosiated with the largest eigen value of that similarity
			matrix. The eigen vecotr is then used as an embedding of that graph.
		"""
		emb_df = pd.DataFrame()
		for g_obj in tqdm(self.graph_c.graph_collection, desc="Loading embeddings"):
			if "embedding" not in g_obj:
				raise ValueError("You have to run node/structural embedding first.")
			if source_node_embedding not in g_obj["embedding"]:
				raise ValueError("No such selected embedding.")
			embs = g_obj["embedding"][source_node_embedding]
			emb_matrix = pd.DataFrame(embs).values
			sim_matrix = cosine_similarity(emb_matrix, emb_matrix)
			eigenvalues, eigenvectors = LA.eig(sim_matrix)
			eigenvalues = [i.real for i in eigenvalues]
			ei_ev_map = {}
			for ii in range(len(eigenvalues)):
				ei_ev_map[eigenvalues[ii]] = eigenvectors[ii]
			max_ei = max(eigenvalues)
			largest_eigen_vector = ei_ev_map[max_ei]
			largest_eigen_vector = [i.real for i in largest_eigen_vector]
			self.sim_matrix_largets_eigen_values.append(max_ei)
			emb_df["emb_"+str(g_obj["graph_id"])] = largest_eigen_vector 
		self.graph_embedding[source_node_embedding] = emb_df.values
		self.graph_embedding_df[source_node_embedding] = emb_df


	def build_wasserstein_graph_embedding(self, source_node_embedding):
		"""
			This method uses the source node mebdding type and builds the graph
			embedding using the Wasserstein method.
			** Note this method does not make sense for classical node embeddings.
		"""
		n = self.graph_c.total_numb_of_nodes
		rows = self.graph_c.graph_id_node_array
		cols = np.arange(n)
		incidence_matrix = scipy.sparse.csr_matrix((np.repeat(1.0,n).astype(np.float32), (rows, cols)))

		embedding_collection = []
		graph_ids = []
		for g_obj in tqdm(self.graph_c.graph_collection, desc="Loading embeddings"):
			if "embedding" not in g_obj:
				raise ValueError("You have to run node/structural embedding first.")
			if source_node_embedding not in g_obj["embedding"]:
				raise ValueError("No such selected embedding.")
			embs = g_obj["embedding"][source_node_embedding]
			embedding_collection.append(list(embs.values()))
			graph_ids.append(g_obj["graph_id"])

		embedding_collection = np.array(embedding_collection, dtype=object)
		embedding_collection = np.vstack(embedding_collection)

		graphs_embed = vectorizers.ApproximateWassersteinVectorizer(
			normalization_power=0.66,
			random_state=42,
		).fit_transform(incidence_matrix.astype(float), vectors=embedding_collection.astype(float))
		self.graph_embedding[source_node_embedding] = graphs_embed
		df = pd.DataFrame(graphs_embed)
		self.emb_cols = ["emb_"+str(i) for i in range(df.shape[1])]
		df.columns = self.emb_cols
		df["graph_id"] = graph_ids
		self.graph_embedding_df[source_node_embedding] = df
		

	@check_gc_status
	def add_graph_labels(self, graph_label_csv_path):
		"""
			This function takes as input a csv file for graph labels and uses the pre-built
			graph collection object, to assign labels to graphs.
		"""
		self.graph_c.assign_graph_labels(graph_label_csv_path)


	@check_gc_status
	def reduce_graph_embedding_dimension(self, algorithm, source_embedding):
		"""
			This function uses the graph embeddings built using source node/structural embedding
			and saves it.
		"""
		# Check if source embedding exists
		if not source_embedding in self.graph_embedding:
			raise ValueError("Missing graph embedding for source: %s" %(source_embedding))
		if source_embedding not in self.graph_emb_dim_reduced:
			self.graph_emb_dim_reduced[source_embedding] = {}
		if algorithm == "umap":
			reducer = umap.UMAP()
			redu_emb = reducer.fit_transform(self.graph_embedding[source_embedding])
			self.graph_emb_dim_reduced[source_embedding]["umap"] = {}
			self.graph_emb_dim_reduced[source_embedding]["umap"]["x"] = redu_emb[:,0]
			self.graph_emb_dim_reduced[source_embedding]["umap"]["y"] = redu_emb[:,1]
		elif algorithm == "tsne":
			reducer = TSNE(n_components=2, random_state=0)
			redu_emb = reducer.fit_transform(self.graph_embedding[source_embedding])
			self.graph_emb_dim_reduced[source_embedding]["tsne"] = {}
			self.graph_emb_dim_reduced[source_embedding]["tsne"]["x"] = redu_emb[:,0]
			self.graph_emb_dim_reduced[source_embedding]["tsne"]["y"] = redu_emb[:,1]
		elif algorithm == "pca":
			reducer = PCA(n_components=2)
			redu_emb = reducer.fit_transform(self.graph_embedding[source_embedding])
			self.graph_emb_dim_reduced[source_embedding]["pca"] = {}
			self.graph_emb_dim_reduced[source_embedding]["pca"]["x"] = redu_emb[:,0]
			self.graph_emb_dim_reduced[source_embedding]["pca"]["y"] = redu_emb[:,1]
		else:
			raise ValueError("Selected algorithm is not supported.")


	@check_gc_status
	def plot_graph_embedding(self, source_embedding, dim_reduc_algo, color_using_grpah_labels=False):
		if source_embedding not in self.graph_emb_dim_reduced:
			raise ValueError("No data found for: %s" %(source_embedding))
		if dim_reduc_algo not in self.graph_emb_dim_reduced[source_embedding]:
			raise ValueError("No data found for: %s" %(dim_reduc_algo))

		if color_using_grpah_labels:
			if not self.graph_c.graph_label_list_unique:
				raise ValueError("You have to set graph labels first.")

			color_numbs = list(range(5, 5*len(self.graph_c.graph_label_list_unique)+1, 5))
			colormap = plt.get_cmap('viridis')
			color_numbs_norm = plt.Normalize(min(color_numbs), max(color_numbs))
			colors = [colormap(color_numbs_norm(value)) for value in color_numbs]

			color_df = pd.DataFrame()
			color_df["graph_label"] = self.graph_c.graph_label_list_unique
			color_df["graph_label_color"] = colors

			color_df = self.graph_c.grpah_labels_df.merge(color_df, on="graph_label")

			x = self.graph_emb_dim_reduced[source_embedding][dim_reduc_algo]["x"]
			y = self.graph_emb_dim_reduced[source_embedding][dim_reduc_algo]["y"]
			colors = color_df["graph_label_color"].tolist()

			plt.figure()
			plt.scatter(x, y, c=colors, cmap='viridis', s=100)
			plt.xlabel('Dim-1', fontsize=12)
			plt.ylabel('Dim-2', fontsize=12)
			plt.title("Low-Dim Representation of Graph Embeddings", fontsize=14)
			plt.show()

		else:
			x = self.graph_emb_dim_reduced[source_embedding][dim_reduc_algo]["x"]
			y = self.graph_emb_dim_reduced[source_embedding][dim_reduc_algo]["y"]
			plt.figure()
			plt.scatter(x, y, cmap='viridis', s=100)
			plt.xlabel('Dim-1', fontsize=12)
			plt.ylabel('Dim-2', fontsize=12)
			plt.title("Low-Dim Representation of Graph Embeddings", fontsize=14)
			plt.show()


	def build_classifier(self, classifier_type, source_embedding):
		if source_embedding not in self.graph_c.built_embeddings:
			raise ValueError("Source embedding %s does not exit." % (built_embeddings))
		data_obj = self.format_data_for_classification(source_embedding)
		classifier_results = self.ml_model.build_classifier(data_obj, classifier_type)
		return classifier_results


	def format_data_for_classification(self, source_embedding):
		graph_emb = self.graph_embedding_df[source_embedding]
		data = self.graph_c.grpah_labels_df.merge(graph_emb, on="graph_id")
		data_obj = {}
		data_obj["data"] = data
		data_obj["x_cols"] = self.emb_cols
		data_obj["y_col"] = "graph_label"
		return data_obj