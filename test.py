from ugaf.ugaf import UGAF


config = {}
config["directed_graphs"] = "no"
config["edge_file"] = "/Users/ash/Desktop/development/research/sandbox/ugaf/bzr_dataset/clean_dataset/edge_file.csv"
config["node_to_graph_mapping"] = "/Users/ash/Desktop/development/research/sandbox/ugaf/bzr_dataset/clean_dataset/node_to_graph_mapping.csv"



ugaf = UGAF()
ugaf.build_graph_collection(config)