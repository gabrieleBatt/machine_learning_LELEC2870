from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

from common import *

class myLinearModel():
	def __init__(self, X_sets, Y_sets, feats):
		self.X_final_training = X_sets[0][feats]
		self.Y_final_training = Y_sets[0]
		self.X_test = X_sets[1][feats]
		self.Y_test = Y_sets[1]
		
		self.X_training = X_sets[2][feats]
		self.Y_training = Y_sets[2]
		self.X_validation = X_sets[3][feats]
		self.Y_validation = Y_sets[3]
	
		
		self.model = LinearRegression().fit(self.X_training.values, self.Y_training.values)
		
	def dumpScore(self):
		dump("Score Linear Model:", self.model.score(self.X_validation.values, self.Y_validation.values))
		return self
	
	def validate(self):
		return RMSE(self.model.predict(self.X_validation.values), self.Y_validation.values)
	
	def test(self):
		self.model = LinearRegression().fit(self.X_final_training.values, self.Y_final_training.values)
		rmse = RMSE(self.model.predict(self.X_test.values), self.Y_test.values)
		return rmse

	
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
	def __init__(self,  X_sets, Y_sets, parameter_set, feats):
		self.X_final_training = X_sets[0][feats]
		self.Y_final_training = Y_sets[0]
		self.X_test = X_sets[1][feats]
		self.Y_test = Y_sets[1]
		
		self.X_training = X_sets[2][feats]
		self.Y_training = Y_sets[2]
		self.X_validation = X_sets[3][feats]
		self.Y_validation = Y_sets[3]

		rmse = 1000
		for hidden_layer_units in parameter_set:
		
			new_model = MLPRegressor((hidden_layer_units,hidden_layer_units), activation='relu', max_iter=10000).fit(self.X_training.values, self.Y_training.values.ravel())
			
			new_rmse = RMSE(new_model.predict(self.X_validation.values), self.Y_validation.values)
			
			if(new_rmse < rmse):
				rmse = new_rmse
				self.parameter = hidden_layer_units
				self.model = new_model

	def dumpScore(self):
		dump("Score MLP:", self.model.score(self.X_test.values, self.Y_test.values))
		return self
	
	def validate(self):
		rmse = RMSE(self.model.predict(self.X_validation.values), self.Y_validation.values)
		return rmse
	
	def test(self):
		self.model = MLPRegressor((self.parameter,), activation='relu', max_iter=100000).fit(self.X_final_training.values, self.Y_final_training.values.ravel())
		rmse = RMSE(self.model.predict(self.X_test.values), self.Y_test.values)
		return rmse
		
	def getModel(self):
		return self.model
		
	def getParameter(self):
		return self.parameter
	