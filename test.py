import os
import pandas as pd
from ugaf.ugaf import UGAF

dataset_name = "PROTEINS"

base_config = {
	"config_name" : "example_1",
	"quiet_mode" : "no",
	"data_files" : {
		"edge_csv_path" : "https://raw.githubusercontent.com/elmspace/ugaf_experiments_data/main/real_world_graphs/"+dataset_name+"/processed_data/edge_file.csv",
		"graph_label_map_csv_path" : "https://raw.githubusercontent.com/elmspace/ugaf_experiments_data/main/real_world_graphs/"+dataset_name+"/processed_data/graph_label_mapping_file.csv",
		"node_graph_map_csv_path" : "https://raw.githubusercontent.com/elmspace/ugaf_experiments_data/main/real_world_graphs/"+dataset_name+"/processed_data/node_graph_mapping_file.csv",
		"node_feature_csv_path" : "https://raw.githubusercontent.com/elmspace/ugaf_experiments_data/main/real_world_graphs/"+dataset_name+"/processed_data/node_feature_file.csv"
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
				"feature_name" : "self_walk",
				"type" : "self_walk",
				"emb_dim" : 8
			},
			{
				"feature_name" : "lsme",
				"type" : "lsme",
				"emb_dim" : 8
			},
			{
				"feature_name" : "basic_expansion",
				"type" : "basic_expansion",
				"emb_dim" : 8
			},
			{
				"feature_name" : "structural_node_feature",
				"type" : "degree_centrality",
				"emb_dim" : 8
			},
			{
				"feature_name" : "structural_node_feature",
				"type" : "closeness_centrality",
				"emb_dim" : 8
			},
			{
				"feature_name" : "structural_node_feature",
				"type" : "load_centrality",
				"emb_dim" : 8
			},
			{
				"feature_name" : "structural_node_feature",
				"type" : "eigenvector_centrality",
				"emb_dim" : 8
			}
		],
		"gloabl_embedding" : {
			"type" : "concat",
			"dim_reduction" : {
				"flag" : "yes",
				"reducer_type" : "pca",
				"emb_dim" : 55
			}
		}
	},
	"graph_embedding" : {
		"embedding_type" : "wasserstein",
		"graph_emb_dim" : "auto",
		"dim_reduction" : {
			"flag" : "no",
			"emb_dim" : 6
		}
	},
	"machine_learning_modelling" : {
		"type" : "classification",
		"sample_size" : 50,
		"balance_data" : "yes"
	}
}


if __name__ == '__main__':

	import time

	ugaf = UGAF(config=base_config, config_type="object")

	ugaf.build_graph_collection()
	ugaf.add_graph_labels()
	ugaf.extract_graph_features()
	ugaf.build_graph_embedding()
	ugaf.build_model()

	res_df = pd.DataFrame()
	res_df["Accuracy"] = ugaf.ml_model_results["accuracy"]
	res_df["Dataset"] = dataset_name
	res_df["Source"] = "NEExT"

	print(res_df["Accuracy"].mean())

	# res_df.to_csv("./tmp_results/"+dataset_name+".csv", index=False)

	# fig, data = ugaf.visualize_graph_embedding(color_by="graph_label")
	# fig.show()



