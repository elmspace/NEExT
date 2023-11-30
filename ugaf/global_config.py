"""

"""

# External libraries
import json

# Internal libraries
from ugaf.singleton_template import Singleton

@Singleton
class Global_Config:

	def __init__(self):
		self.config = None


	def load_config(self, config_file_path):
		"""
			This method will simply load the global configuration
			file.
		"""
		print(config_file_path)
		with open(config_file_path, "r") as config_file:
			self.config = dict(json.load(config_file))