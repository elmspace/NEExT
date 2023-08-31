from ugaf.ugaf import UGAF


config = {}
config["g1_path"] = "/Users/ash/Desktop/development/research/testing/ugaf/data/basic_random_graphs/lsme/g_0.csv"
config["g2_path"] = "/Users/ash/Desktop/development/research/testing/ugaf/data/basic_random_graphs/lsme/g_0.csv"
config["compute_wasserstein_distance"] = True
config["compute_similatity_matrix_eigen_values"] = True
config["compute_similarity_matrix_distribution_properties"] = True

config["sample_rate"] = 1.0
config["ensemble_size"] = 1


obj = UGAF(config)
obj.run()