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
		with open(config_file_path) as config_file:
			self.config = dict(json.load(config_file))