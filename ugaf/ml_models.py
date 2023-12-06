"""
	Author : Ash Dehghan
	Description: 
"""

# External Libraries
import xgboost
from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Internal Modules
from ugaf.global_config import Global_Config

class ML_Models:


	def __init__(self):
		self.global_config = Global_Config.instance()


	def build_model(self, data_obj):
		model_type = self.global_config.config["machine_learning_modelling"]["type"]
		if model_type == "regression":
			model_result = self.run_regression_models(data_obj)
		elif model_type == "classification":
			model_result = self.run_classification_models(data_obj)
		else:
			raise ValueError("Model type not supported.")
		return model_result


	def run_classification_models(self, data_obj):
		sample_size = self.global_config.config["machine_learning_modelling"]["sample_size"]
		result = {}
		result["accuracy"] = []
		result["precision"] = []
		result["recall"] = []
		result["f1"] = []
		for i in tqdm(range(sample_size), desc="Building models:", disable=self.global_config.quiet_mode):
			data_obj = self.format_data(data_obj, format_type="classification")
			accuracy, precision, recall, f1 = self.build_xgboost_classification(data_obj)
			result["accuracy"].append(accuracy)
			result["precision"].append(precision)
			result["recall"].append(recall)
			result["f1"].append(f1)
		return result


	def run_regression_models(self, data_obj):
		sample_size = self.global_config.config["machine_learning_modelling"]["sample_size"]
		result = {}
		result["mse"] = []
		result["mae"] = []
		for i in tqdm(range(sample_size), desc="Building models:", disable=self.global_config.quiet_mode):
			data_obj = self.format_data(data_obj)
			mse, mae = self.build_xgboost_regression(data_obj)
			result["mse"].append(mse)
			result["mae"].append(mae)
		return result


	def build_xgboost_classification(self, data_obj):
		model = xgboost.XGBClassifier(n_estimators=1000, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8)
		model.fit(data_obj["X_train"], data_obj["y_train"])
		y_pred = model.predict(data_obj["X_test"]).flatten()
		y_true = data_obj["y_test"]
		accuracy = accuracy_score(y_true, y_pred)
		precision = precision_score(y_true, y_pred, average='micro')
		recall = recall_score(y_true, y_pred, average='micro')
		f1 = f1_score(y_true, y_pred, average='micro')
		return accuracy, precision, recall, f1


	def build_xgboost_regression(self, data_obj):
		model = xgboost.XGBRegressor(n_estimators=1000, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8)
		model.fit(data_obj["X_train"], data_obj["y_train"])
		y_pred = model.predict(data_obj["X_test"]).flatten()
		y_true = data_obj["y_test"]
		mse = mean_squared_error(y_true, y_pred)
		mae = mean_absolute_error(y_true, y_pred)
		return mse, mae


	def format_data(self, data_obj, format_type="regression"):
		"""
			This function will take the raw data object and will create a 
			normalized train, test and validation sets.
		"""
		df = data_obj["data"].copy(deep=True)
		df = df.sample(frac=1).copy(deep=True)
		
		if format_type == "classification":
			# We need to relabel the classes to fit XGBoost format
			raw_classes = list(set(df[[data_obj["y_col"]]]["graph_label"]))
			class_map = {}
			class_remap = 0
			for class_val in raw_classes:
				class_map[class_val] = class_remap
				class_remap += 1

			df[data_obj["y_col"]] = df[data_obj["y_col"]].apply(lambda x : class_map[x])

		X = df[data_obj["x_cols"]]
		y = df[[data_obj["y_col"]]]
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
		X_train, X_vald, y_train, y_vald = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
		# Standardize data
		scaler = StandardScaler()
		scaler.fit(X_train)
		X_train = scaler.fit_transform(X_train)
		X_test = scaler.fit_transform(X_test)
		X_vald = scaler.fit_transform(X_vald)
		data_obj["X_train"] = X_train
		data_obj["X_test"] = X_test
		data_obj["X_vald"] = X_vald
		data_obj["y_train"] = y_train.values
		data_obj["y_test"] = y_test.values
		data_obj["y_vald"] = y_vald.values
		return data_obj



