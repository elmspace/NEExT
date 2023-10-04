"""
	Author : Ash Dehghan
	Description: This class uses the Networkx library as a base to
	build a class that handles a collection of graphs, to be used
	for graph analysis.
"""


import pandas as pd
import networkx as nx
from tqdm import tqdm
from loguru import logger


class Graph_Collection:

	def __init__(self):
		self.gc_status = False
		self.graph_collection = []



	def load_graphs(self, edge_csv_path, node_graph_map_csv_path):
		"""
			This method uses the user configuration to build a collection
			of graphs object.
		"""
		logger.info("===================")
		logger.info("Rading edge file")

		edges = pd.read_csv(edge_csv_path)
		src_nodes = [int(i) for i in edges["node_a"].tolist()]
		dst_nodes = [int(i) for i in edges["node_b"].tolist()]
		edgelist = list(zip(src_nodes, dst_nodes))
		G = nx.from_edgelist(edgelist)

		logger.info("Parsing graph data into individual graphs")

		node_graph_map = pd.read_csv(node_graph_map_csv_path)
		node_graph_map["node_id"] = node_graph_map["node_id"].astype(int)
		node_graph_map["graph_id"] = node_graph_map["graph_id"].astype(int)
		graph_ids = node_graph_map["graph_id"].unique().tolist()
		for graph_id in tqdm(graph_ids, desc="Building subgraphs:"):
			node_list = node_graph_map[node_graph_map["graph_id"] == graph_id]["node_id"].tolist()
			g = nx.Graph(G.subgraph(node_list))
			cc = list(nx.connected_components(g))
			g_obj = {}
			g_obj["graph"] = g
			g_obj["numb_of_nodes"] = len(g.nodes)
			g_obj["numb_of_edges"] = len(g.edges)
			g_obj["numb_of_connected_components"] = len(cc)
			g_obj["connected_components"] = sorted(cc, key=len, reverse=True)
			g_obj["graph_id"] = graph_id
			self.graph_collection.append(g_obj)

		self.gc_status = True



	def filter_collection_for_largest_connected_component(self):
		"""
			This method will go through all the sub-graphs and if the number
			of component of the sub-graph is greater than 1, it will only keep the largest component.
		"""
		# Check to see if graph collection was built
		if not self.gc_status:
			logger.error("You need to build a graph collection first.")

		logger.info("===================")
		logger.info("Filtering graphs for to contain only the largest connected component")

		for g_obj in tqdm(self.graph_collection, desc="Filtering graphs:"):
			largest_cc = list(g_obj["connected_components"][0])
			g = nx.Graph(g_obj["graph"].subgraph(largest_cc))
			cc = list(nx.connected_components(g))
			g_obj["graph"] = g
			g_obj["numb_of_nodes"] = len(g.nodes)
			g_obj["numb_of_edges"] = len(g.edges)
			g_obj["numb_of_connected_components"] = len(cc)
			g_obj["connected_components"] = sorted(cc, key=len, reverse=True)



	def reset_node_indices(self):
		"""
			This method will reset the node indices to start from 0.
			It will keep a mapping between old and new node indices.
		"""
		for g_obj in tqdm(self.graph_collection, desc="Resrting node indices:"):
			g = g_obj["graph"]
			mapping = {}
			current_nodes = list(g.nodes)
			for idx, node in enumerate(current_nodes):
				mapping[node] = idx
			g_obj["graph"] = nx.relabel_nodes(g, mapping)
			g_obj["re_index_map"] = mapping
