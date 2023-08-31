from ugaf.ugaf import UGAF


config = {}
config["g1_path"] = "/Users/ash/Desktop/development/research/testing/ugaf/mix_1.csv"
config["g2_path"] = "/Users/ash/Desktop/development/research/testing/ugaf/mix_2.csv"
config["sample_rate"] = 0.05
config["ensemble_size"] = 10


obj = UGAF(config)
obj.run()