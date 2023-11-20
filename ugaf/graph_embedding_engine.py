"""
	Author: Ash Dehghan
"""
import scipy
import random
import vectorizers
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm


class Graph_Embedding_Engine:


	def __init__(self):
		pass


	def build_graph_embedding(self, graph_embedding_type, graph_c):

		if graph_embedding_type == "wasserstein":
			graphs_embed, graph_embedding_df = self.build_wasserstein_graph_embedding(graph_c)
		else:
			raise ValueError("Graph embedding type selected is not supported.")
		return graphs_embed, graph_embedding_df


	def build_wasserstein_graph_embedding(self, graph_c):
		"""
			This method uses the source node mebdding type and builds the graph
			embedding using the Wasserstein method.
			** Note this method does not make sense for classical node embeddings.
		"""
		n = graph_c.total_numb_of_nodes
		rows = graph_c.graph_id_node_array
		cols = np.arange(n)
		incidence_matrix = scipy.sparse.csr_matrix((np.repeat(1.0,n).astype(np.float32), (rows, cols)))
		
		embedding_collection = []
		graph_ids = []
		for g_obj in tqdm(graph_c.graph_collection, desc="Loading embeddings"):
			embs = g_obj["graph_features"]["global_embedding"]
			embedding_collection.append(list(embs.values()))
			graph_ids.append(g_obj["graph_id"])

		embedding_collection = np.array(embedding_collection, dtype=object)
		embedding_collection = np.vstack(embedding_collection)
		graphs_embed = vectorizers.ApproximateWassersteinVectorizer(
			normalization_power=0.66,
			random_state=42,
		).fit_transform(incidence_matrix.astype(float), vectors=embedding_collection.astype(float))
		graph_embedding_df = pd.DataFrame(graphs_embed)
		emb_cols = ["emb_"+str(i) for i in range(graph_embedding_df.shape[1])]
		graph_embedding_df.columns = emb_cols
		graph_embedding_df["graph_id"] = graph_ids
		return graphs_embed, graph_embedding_df
