from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
import pandas as pd

from common import *

class myLinearModel():
	def __init__(self, X_test_training, Y_test_training, feats):
		self.feats = feats
		self.model = LinearRegression().fit(X_test_training[self.feats], Y_test_training)
		
	def predict(self, X):
		return self.model.predict(X[self.feats])

	
class myKNN():
	def __init__(self, X_sets, Y_sets, k_set, feats):
		self.X_final_training = X_sets[0][feats]
		self.Y_final_training = Y_sets[0]
		self.X_test = X_sets[1][feats]
		self.Y_test = Y_sets[1]
		
		self.X_training = X_sets[2][feats]
		self.Y_training = Y_sets[2]
		self.X_validation = X_sets[3][feats]
		self.Y_validation = Y_sets[3]
	
		rmse = 1000
		for n_n in k_set:
		
			new_model = KNeighborsRegressor(n_neighbors=n_n).fit(self.X_training.values, self.Y_training.values)
			
			new_rmse = RMSE(new_model.predict(self.X_validation.values), self.Y_validation.values)
			
			if(new_rmse < rmse):
				rmse = new_rmse
				self.k = n_n
				self.model = new_model
	

	def dumpScore(self):
		dump("Score KNN:", self.model.score(self.X_test.values, self.Y_test.values))
		return self
	
	def validate(self):
		rmse = RMSE(self.model.predict(self.X_validation.values), self.Y_validation.values)
		return rmse
	
	def test(self):
		self.model = KNeighborsRegressor(n_neighbors=self.k).fit(self.X_final_training.values, self.Y_final_training.values)
		rmse = RMSE(self.model.predict(self.X_test.values), self.Y_test.values)
		return rmse
		
	def getModel(self):
		return self.model
		
	def getK(self):
		return self.k
	

class myMLP():
	def __init__(self,  X_test_training, Y_test_training, X_validation_training, Y_validation_training, X_validation, Y_validation, parameter_set, feats):
		self.feats = feats
		if(len(parameter_set) > 1):
			rmse = 1000
			for hidden_layer_units in parameter_set:
			
				new_model = MLPRegressor((hidden_layer_units,hidden_layer_units), activation='relu', max_iter=10000).fit(X_validation_training[self.feats].values, Y_validation_training.values.ravel())
				
				new_rmse = RMSE(new_model.predict(X_validation.values), Y_validation.values)
				
				if(new_rmse < rmse):
					rmse = new_rmse
					self.parameter = hidden_layer_units
		else:
			self.parameter = parameter_set[0]
			
		self.model = MLPRegressor((self.parameter,), activation='relu', max_iter=100000).fit(X_test_training[self.feats].values, Y_test_training.values.ravel())
				
	def predict(self, X):
		self.model.predict(X[self.feats])
	