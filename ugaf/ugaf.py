import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
# Tensorflo Modules
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model


class UGAF:

	def __init__(self, config):
		self.config = config
		self.g1_emb = pd.DataFrame()
		self.g2_emb = pd.DataFrame()
		self.emb_samps = {"g1":[], "g2":[]}
		self.sim_matrices = {"g1vg1":[], "g2vg2":[], "g1vg2":[]}
		self.stats = {}


	def run(self):
		self.load_data()
		self.sample_graphs()
		self.build_similarity_matrix()
		self.compute_wasserstein_distance()
		self.compute_similatity_matrix_eigen_values()
		self.compute_similarity_matrix_distribution_properties()


	def load_data(self):
		"""
			This method loads the embedding data for graph 1 and 2.
		"""
		# Load embedding data
		try:
			self.g1_emb = pd.read_csv(self.config["g1_path"])
			self.g2_emb = pd.read_csv(self.config["g2_path"])
		except Exception as e:
			print("Failed to load data. See error:")
			print(str(e))


	def sample_graphs(self):
		"""
			This method samples the two graph embedding based on the sample rate defined by the user.
		"""
		sample_size = int(self.config["sample_rate"] * min([len(self.g1_emb), len(self.g2_emb)]))
		for i in tqdm(range(self.config["ensemble_size"]), desc="Sampling Embeddings:"):
			self.emb_samps["g1"].append(self.g1_emb.sample(n=sample_size))
			self.emb_samps["g2"].append(self.g2_emb.sample(n=sample_size))


	def build_similarity_matrix(self):
		"""
			This method calculates similarity measure between the two sample
			graph embeddings.
		"""
		# g1 sims
		for i in tqdm(range(len(self.emb_samps["g1"])), desc="Similarity Measure g1vg1"):
			for j in range(i+1, len(self.emb_samps["g1"])):
				e1 = self.emb_samps["g1"][i]
				e2 = self.emb_samps["g1"][j]
				self.sim_matrices["g1vg1"].append(cosine_similarity(e1.values, e2.values))
		
		# g2 sims
		for i in tqdm(range(len(self.emb_samps["g2"])-1), desc="Similarity Measure g2vg2"):
			for j in range(i+1, len(self.emb_samps["g2"])):
				e1 = self.emb_samps["g2"][i]
				e2 = self.emb_samps["g2"][j]
				self.sim_matrices["g2vg2"].append(cosine_similarity(e1.values, e2.values))
		
		# g1 v g2 sims
		for i in tqdm(range(len(self.emb_samps["g1"])), desc="Similarity Measure g1vg2"):
			for j in range(i, len(self.emb_samps["g2"])):
				e1 = self.emb_samps["g1"][i]
				e2 = self.emb_samps["g2"][j]
				self.sim_matrices["g1vg2"].append(cosine_similarity(e1.values, e2.values))


	def compute_wasserstein_distance(self):
		"""
			This method calculates the Wasserstein Distance between
			similarity matrices.
		"""
		# G1vG1 Measure
		was_dist_g1vg1 = []
		for i in tqdm(range(len(self.sim_matrices["g1vg1"])), desc="Measuring Wasserstein Distance g1vg1"):
			for j in range(i+1, len(self.sim_matrices["g1vg1"])):
				s1 = self.sim_matrices["g1vg1"][i].flatten()
				s2 = self.sim_matrices["g1vg1"][j].flatten()
				was_dist = wasserstein_distance(s1, s2)
				was_dist_g1vg1.append(was_dist)
		was_dist_g1vg1 = np.array(was_dist_g1vg1)
		# G2vG2 Measure
		was_dist_g2vg2 = []
		for i in tqdm(range(len(self.sim_matrices["g2vg2"])-1), desc="Measuring Wasserstein Distance g2vg2"):
			for j in range(i+1, len(self.sim_matrices["g2vg2"])):
				s1 = self.sim_matrices["g2vg2"][i].flatten()
				s2 = self.sim_matrices["g2vg2"][j].flatten()
				was_dist = wasserstein_distance(s1, s2)
				was_dist_g2vg2.append(was_dist)
		was_dist_g2vg2 = np.array(was_dist_g2vg2)
		# G1vG2 Measure
		was_dist_g1vg2 = []
		for i in tqdm(range(len(self.sim_matrices["g1vg2"])-1), desc="Measuring Wasserstein Distance g1vg2"):
			for j in range(i+1, len(self.sim_matrices["g1vg2"])):
				s1 = self.sim_matrices["g1vg2"][i].flatten()
				s2 = self.sim_matrices["g1vg2"][j].flatten()
				was_dist = wasserstein_distance(s1, s2)
				was_dist_g1vg2.append(was_dist)
		was_dist_g1vg2 = np.array(was_dist_g1vg2)
		# Set metrics
		self.stats["wasserstein_distance"] = {}
		self.stats["wasserstein_distance"]["g1vg1"] = was_dist_g1vg1
		self.stats["wasserstein_distance"]["g2vg2"] = was_dist_g2vg2
		self.stats["wasserstein_distance"]["g1vg2"] = was_dist_g1vg2


	def compute_similatity_matrix_eigen_values(self):
		"""
			This method computes the eigen values of the similarity matrix.
		"""
		self.stats["sim_matrix_eigen_values"] = {}
		self.stats["sim_matrix_eigen_values"]["g1vg1"] = {"eigen_values":[]}
		self.stats["sim_matrix_eigen_values"]["g2vg2"] = {"eigen_values":[]}
		self.stats["sim_matrix_eigen_values"]["g1vg2"] = {"eigen_values":[]}
		# Compute eigen values for g1vg1
		eigen_g1vg1 = []
		for i in tqdm(range(len(self.sim_matrices["g1vg1"])), desc="Eigen values of g1vg1"):
			val = np.linalg.eig(self.sim_matrices["g1vg1"][i])
			val = val[0][0].real
			eigen_g1vg1.append(val)
		self.stats["sim_matrix_eigen_values"]["g1vg1"]["eigen_values"] = eigen_g1vg1
		# Compute eigen values for g2vg2
		eigen_g2vg2 = []
		for i in tqdm(range(len(self.sim_matrices["g2vg2"])), desc="Eigen values of g2vg2"):
			val = np.linalg.eig(self.sim_matrices["g2vg2"][i])
			val = val[0][0].real
			eigen_g2vg2.append(val)
		self.stats["sim_matrix_eigen_values"]["g2vg2"]["eigen_values"] = eigen_g2vg2
		# Compute eigen values for g1vg2
		eigen_g1vg2 = []
		for i in tqdm(range(len(self.sim_matrices["g1vg2"])), desc="Eigen values of g1vg2"):
			val = np.linalg.eig(self.sim_matrices["g1vg2"][i])
			val = val[0][0].real
			eigen_g1vg2.append(val)
		self.stats["sim_matrix_eigen_values"]["g1vg2"]["eigen_values"] = eigen_g1vg2


	def compute_similarity_matrix_distribution_properties(self):
		"""
			This method computes various statistical properties of the 
		"""
		self.stats["sim_matrix_dist_properties"] = {}
		self.stats["sim_matrix_dist_properties"]["g1vg1"] = {"kurtosis":[], "mean":[], "std":[], "skew":[], "cov":[]}
		self.stats["sim_matrix_dist_properties"]["g2vg2"] = {"kurtosis":[], "mean":[], "std":[], "skew":[], "cov":[]}
		self.stats["sim_matrix_dist_properties"]["g1vg2"] = {"kurtosis":[], "mean":[], "std":[], "skew":[], "cov":[]}
		# Run stats g1vg1
		for mtx in tqdm(self.sim_matrices["g1vg1"], desc="Computing sim-matrix properties g1vg1:"):
			mtx_flat = mtx.flatten()
			self.stats["sim_matrix_dist_properties"]["g1vg1"]["kurtosis"].append(kurtosis(mtx_flat))
			self.stats["sim_matrix_dist_properties"]["g1vg1"]["mean"].append(mtx_flat.mean())
			self.stats["sim_matrix_dist_properties"]["g1vg1"]["std"].append(np.std(mtx_flat))
			self.stats["sim_matrix_dist_properties"]["g1vg1"]["skew"].append(skew(mtx_flat))
			self.stats["sim_matrix_dist_properties"]["g1vg1"]["cov"].append(np.std(mtx_flat)/mtx_flat.mean())
		# Run stats g2vg2
		for mtx in tqdm(self.sim_matrices["g2vg2"], desc="Computing sim-matrix properties g2vg2:"):
			mtx_flat = mtx.flatten()
			self.stats["sim_matrix_dist_properties"]["g2vg2"]["kurtosis"].append(kurtosis(mtx_flat))
			self.stats["sim_matrix_dist_properties"]["g2vg2"]["mean"].append(mtx_flat.mean())
			self.stats["sim_matrix_dist_properties"]["g2vg2"]["std"].append(np.std(mtx_flat))
			self.stats["sim_matrix_dist_properties"]["g2vg2"]["skew"].append(skew(mtx_flat))
			self.stats["sim_matrix_dist_properties"]["g2vg2"]["cov"].append(np.std(mtx_flat)/mtx_flat.mean())
		# Run stats g1vg2
		for mtx in tqdm(self.sim_matrices["g1vg2"], desc="Computing sim-matrix properties g1vg2:"):
			mtx_flat = mtx.flatten()
			self.stats["sim_matrix_dist_properties"]["g1vg2"]["kurtosis"].append(kurtosis(mtx_flat))
			self.stats["sim_matrix_dist_properties"]["g1vg2"]["mean"].append(mtx_flat.mean())
			self.stats["sim_matrix_dist_properties"]["g1vg2"]["std"].append(np.std(mtx_flat))
			self.stats["sim_matrix_dist_properties"]["g1vg2"]["skew"].append(skew(mtx_flat))
			self.stats["sim_matrix_dist_properties"]["g1vg2"]["cov"].append(np.std(mtx_flat)/mtx_flat.mean())











