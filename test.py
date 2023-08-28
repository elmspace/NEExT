from ugaf.ugaf import UGAF


config = {}
config["g1_path"] = "/Users/ash/Desktop/development/research/testing/ugaf/G1.csv"
config["g2_path"] = "/Users/ash/Desktop/development/research/testing/ugaf/G3.csv"
config["sample_rate"] = 0.2


obj = UGAF(config)
obj.run()