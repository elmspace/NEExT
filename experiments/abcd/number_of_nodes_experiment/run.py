import os
import pandas as pd
from tqdm import tqdm
from ugaf.ugaf import UGAF

base_config = {
	"config_name" : "example_1",
	"quiet_mode" : "no",
	"data_files" : {
		"edge_csv_path" : "https://raw.githubusercontent.com/elmspace/ugaf_experiments_data/main/abcd/xi_n/edge_file.csv",
		"node_graph_map_csv_path" : "https://raw.githubusercontent.com/elmspace/ugaf_experiments_data/main/abcd/xi_n/node_graph_mapping_file.csv",
		"graph_label_map_csv_path" : "https://raw.githubusercontent.com/elmspace/ugaf_experiments_data/main/abcd/xi_n/graph_label_mapping_file.csv"
	},
	"graph_collection" : {
		"filter_for_largest_cc" : "yes",
		"reset_node_indices" : "yes"
	},
	"graph_sample" : {
			"flag" : "no",
			"sample_fraction" : 0.1
	},
	"graph_features" : {
		"features" : [
			{
				"feature_name" : "structural_node_feature",
				"type" : "degree_centrality",
				"emb_dim" : 4
			}
		],
		"gloabl_embedding" : {
			"type" : "concat"
		}
	},
	"graph_embedding" : {
		"embedding_type" : "wasserstein",
		"graph_emb_dim" : 4,
		"dim_reduction" : {
			"flag" : "no",
			"emb_dim" : 8
		}
	},
	"machine_learning_modelling" : {
		"type" : "regression",
		"sample_size" : 50
	}
}

ugaf = UGAF(config=base_config, config_type="object")
ugaf.build_graph_collection()
ugaf.add_graph_labels()
ugaf.extract_graph_features()
ugaf.build_graph_embedding()

fig, data = ugaf.visualize_graph_embedding(color_by="graph_label")
data.to_csv("./results/results.csv", index=False)