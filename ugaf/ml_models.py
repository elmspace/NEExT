"""
	Author : Ash Dehghan
	Description: 
"""

import xgboost
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class ML_Models:


	def __init__(self):
		pass


	def build_model(self, data_obj, model_type):
		
		if model_type == "regression":
			model_result = self.run_regression_models(data_obj)
		elif model_type == "classification":
			pass
		else:
			raise ValueError("Model type not supported.")


	def run_regression_models(self, data_obj):

		for i in range(10):
			data_obj = self.format_data(data_obj)
			mse = self.build_xgboost_regression(data_obj)
			print(mse)


	def build_xgboost_regression(self, data_obj):
		model = xgboost.XGBRegressor(n_estimators=1000, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8)
		model.fit(data_obj["X_train"], data_obj["y_train"])
		y_pred = model.predict(data_obj["X_test"]).flatten()
		y_true = data_obj["y_test"]
		mse = mean_squared_error(y_true, y_pred)
		return mse




	def format_data(self, data_obj):
		"""
			This function will take the raw data object and will create a 
			normalized train, test and validation sets.
		"""
		df = data_obj["data"].copy(deep=True)
		df = df.sample(frac=1).copy(deep=True)
		X = df[data_obj["x_cols"]]
		y = df[[data_obj["y_col"]]]
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
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


	def build_decision_tree_classifier(self, data_obj):
		clf = DecisionTreeClassifier(random_state=0, max_depth=5)
		clf.fit(data_obj["X_train"], data_obj["y_train"])
		y_pred = clf.predict(data_obj["X_test"])
		y_test = data_obj["y_test"]
		performance_report = self.run_performance(y_test, y_pred)
		return performance_report


	def run_performance(self, y_test, y_pred):
		accuracy = accuracy_score(y_test, y_pred)
		precision = precision_score(y_test, y_pred)
		recall = recall_score(y_test, y_pred)
		f1 = f1_score(y_test, y_pred, average='macro')
		performance_report = {}
		performance_report["accuracy"] = accuracy
		performance_report["precision"] = precision
		performance_report["recall"] = recall
		performance_report["f1"] = f1
		return performance_report

