import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class UGAF:

	def __init__(self, config):
		self.config = config
		self.g1_emb = pd.DataFrame()
		self.g2_emb = pd.DataFrame()
		self.g1_emb_samp = pd.DataFrame()
		self.g2_emb_samp = pd.DataFrame()


	def run(self):
		self.load_data()
		self.sample_graphs()
		self.build_similarity_matrix()


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
		random_state_val = None
		if "sample_random_state" in self.config:
			random_state_val = self.config["sample_random_state"]		
		self.g1_emb_samp = self.g1_emb.sample(n=sample_size, random_state=random_state_val)
		self.g2_emb_samp = self.g2_emb.sample(n=sample_size, random_state=random_state_val)


	def build_similarity_matrix(self):
		"""
			This method calculates similarity measure between the two sample
			graph embeddings.
		"""


		sim_matrix = cosine_similarity(self.g1_emb_samp.values, self.g2_emb_samp.values)
		print(sim_matrix)



