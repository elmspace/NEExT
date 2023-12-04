import os
from ugaf.ugaf import UGAF

base_config = {
	"config_name" : "example_1",
	"quiet_mode" : "no",
	"data_files" : {
		"edge_csv_path" : "/Users/ash/Desktop/share/data/ugaf_experiments/abcd/xi/edge_file.csv",
		"node_graph_map_csv_path" : "/Users/ash/Desktop/share/data/ugaf_experiments/abcd/xi/node_graph_mapping_file.csv",
		"graph_label_map_csv_path" : "/Users/ash/Desktop/share/data/ugaf_experiments/abcd/xi/graph_label_mapping_file.csv"
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
				"feature_name" : "lsme",
				"type" : "lsme",
				"emb_dim" : 2
			}
		],
		"gloabl_embedding" : {
			"type" : "concat"
		}
	},
	"graph_embedding" : {
		"embedding_type" : "wasserstein",
		"graph_emb_dim" : 2,
		"dim_reduction" : {
			"flag" : "no",
			"emb_dim" : 8
		}
	},
	"machine_learning_modelling" : {
		"type" : "regression",
		"sample_size" : 10
	}
}




feat_emb_dim = [1, 4, 6, 8, 10]

feat_types = []
feat_types.append([{"feature_name" : "basic_expansion", "type" : "basic_expansion", "emb_dim" : 2}])

for feat in feat_types:
    for dim in feat_emb_dim:

        ugaf = UGAF(config=base_config, config_type="object")

        # Update the config
        ugaf.global_config.config["graph_features"]["features"] = feat
        for x in ugaf.global_config.config["graph_features"]["features"]:
            x["emb_dim"] = dim


        ugaf.build_graph_collection()
        ugaf.add_graph_labels()
        ugaf.extract_graph_features()
        ugaf.build_graph_embedding()
        ugaf.compute_similarity_matrix_stats()
        ugaf.build_model()

        exit(0)
