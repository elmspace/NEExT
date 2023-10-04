from ugaf.ugaf import UGAF


base_dir = "/home/ash/Desktop/development/research/graph_sandbox/ugaf/bzr_dataset/clean_dataset/"

ugaf = UGAF()


edge_csv_path = base_dir + "edge_file.csv"
node_graph_map_csv_path = base_dir + "node_graph_mapping_file.csv"

ugaf.build_graph_collection(edge_csv_path, node_graph_map_csv_path, filter_for_largest_cc=True, reset_node_indices=True)

ugaf.build_node_embedding(embedding_type="node2vec")