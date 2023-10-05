import os
from loguru import logger
from ugaf.ugaf import UGAF


run_location = os.environ["run_location"]

if run_location == "system76":
	base_dir = "/home/ash/Desktop/share/research/ugaf_data/bzr_dataset/"
elif run_location == "home_linux_server":
	base_dir = "/run/user/1000/gvfs/smb-share:server=pop-os.local,share=share/research/ugaf_data/bzr_dataset/"
else:
	logger.error("Wrong run location.")


ugaf = UGAF()


edge_csv_path = base_dir + "edge_file.csv"
node_graph_map_csv_path = base_dir + "node_graph_mapping_file.csv"

ugaf.build_graph_collection(edge_csv_path, node_graph_map_csv_path, filter_for_largest_cc=True, reset_node_indices=True)

## For testing only
ugaf.graph_c.graph_collection = ugaf.graph_c.graph_collection[0:1]

ugaf.build_node_embedding(embedding_type="node2vec")