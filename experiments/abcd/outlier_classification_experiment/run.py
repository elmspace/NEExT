import os
import pandas as pd
from tqdm import tqdm
from ugaf.ugaf import UGAF

base_config = {
	"config_name" : "example_1",
	"quiet_mode" : "no",
	"data_files" : {
		"edge_csv_path" : "https://raw.githubusercontent.com/elmspace/ugaf_experiments_data/main/abcd/outlier_lite/info/edge_file.csv",
		"node_graph_map_csv_path" : "https://raw.githubusercontent.com/elmspace/ugaf_experiments_data/main/abcd/outlier_lite/info/node_graph_mapping_file.csv",
		"graph_label_map_csv_path" : "https://raw.githubusercontent.com/elmspace/ugaf_experiments_data/main/abcd/outlier_lite/info/graph_label_mapping_file.csv"
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
				"feature_name" : "basic_expansion",
				"type" : "basic_expansion",
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
		"type" : "classification",
		"sample_size" : 100
	}
}


feat_types = []
feat_types.append([{"feature_name" : "basic_expansion", "type" : "basic_expansion", "emb_dim" : 4}])
feat_types.append([{"feature_name" : "lsme", "type" : "lsme", "emb_dim" : 4}])
feat_types.append([{"feature_name" : "structural_node_feature", "type" : "page_rank", "emb_dim" : 4}])
feat_types.append([{"feature_name" : "structural_node_feature", "type" : "degree_centrality", "emb_dim" : 4}])
feat_types.append([{"feature_name" : "structural_node_feature", "type" : "closeness_centrality", "emb_dim" : 4}])
feat_types.append([{"feature_name" : "structural_node_feature", "type" : "eigenvector_centrality", "emb_dim" : 4}])
feat_types.append([{"feature_name" : "basic_expansion", "type" : "basic_expansion", "emb_dim" : 4}, {"feature_name" : "lsme", "type" : "lsme", "emb_dim" : 4}])
feat_types.append([{"feature_name" : "basic_expansion", "type" : "basic_expansion", "emb_dim" : 4}, {"feature_name" : "lsme", "type" : "lsme", "emb_dim" : 4}, {"feature_name" : "structural_node_feature", "type" : "page_rank", "emb_dim" : 4}])


results = pd.DataFrame()

for feat in tqdm(feat_types, desc="Features:"):

	ugaf = UGAF(config=base_config, config_type="object")

	# Update the config
	ugaf.global_config.config["graph_features"]["features"] = feat
	run_feat_name = ""
	for x in ugaf.global_config.config["graph_features"]["features"]:
		run_feat_name += x["type"] + " "
	run_feat_name = run_feat_name.strip()        

	ugaf.build_graph_collection()
	ugaf.add_graph_labels()
	ugaf.extract_graph_features()
	ugaf.build_graph_embedding()
	ugaf.build_model()

	df = pd.DataFrame()
	df["accuracy"] = ugaf.ml_model_results["accuracy"]
	df["precision"] = ugaf.ml_model_results["precision"]
	df["recall"] = ugaf.ml_model_results["recall"]
	df["f1"] = ugaf.ml_model_results["f1"]
	df["feat_name"] = run_feat_name

	print(df)
	exit(0)

	if results.empty:
		results = df.copy(deep=True)
	else:
		results = pd.concat([results, df])
		

results.to_csv("./results.csv", index=False)
