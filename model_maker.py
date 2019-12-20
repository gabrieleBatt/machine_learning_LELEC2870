from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

from common import *

class myLinearModel():
	def __init__(self, X_sets, Y_sets, feats):
		self.X_test_training = X_sets[0][feats]
		self.Y_test_training = Y_sets[0]
		self.X_test = X_sets[1][feats]
		self.Y_test = Y_sets[1]
		
		self.X_validation_training = X_sets[2][feats]
		self.Y_validation_training = Y_sets[2]
		self.X_validation = X_sets[3][feats]
		self.Y_validation = Y_sets[3]
	
	def test(self):
		self.model = LinearRegression().fit(self.X_test_training.values, self.Y_test_training.values)
		prediction_set = self.model.predict(self.X_test.values)
		rmse = RMSE(prediction_set, self.Y_test.values)
		return (rmse, prediction_set)

	
class myKNN():
	def __init__(self, X_sets, Y_sets, k_set, feats):
		self.feats = feats
		self.X_test_training = X_sets[0][feats]
		self.Y_test_training = Y_sets[0]
		self.X_test = X_sets[1][feats]
		self.Y_test = Y_sets[1]
		
		self.X_validation_training= X_sets[2][feats]
		self.Y_validation_training = Y_sets[2]
		self.X_validation = X_sets[3][feats]
		self.Y_validation = Y_sets[3]
	
		self.rmse_list = []
		rmse = 1000
		dump("Parameters to validate:", (k_set[0],k_set[-1]))
		bar = Bar(len(k_set))
		for n_n in k_set:
		
			new_model = KNeighborsRegressor(n_neighbors=n_n,weights='distance').fit(self.X_validation_training.values, self.Y_validation_training.values)
			
			new_rmse = RMSE(new_model.predict(self.X_validation.values), self.Y_validation.values)
			
			self.rmse_list.append(new_rmse)
			
			if(new_rmse < rmse):
				rmse = new_rmse
				self.k = n_n
		
			bar.moreBar()
			
		bar.noBar()
		
		dump("K:", self.k)
	
	def test(self):
		self.model = KNeighborsRegressor(n_neighbors=self.k).fit(self.X_test_training.values, self.Y_test_training.values)
		prediction_set = self.model.predict(self.X_test.values)
		rmse = RMSE(prediction_set, self.Y_test.values)
		return (rmse, prediction_set)
		
	def getHistory(self):
		return self.rmse_list
		
	def predict(self, X):
		return self.model.predict(X[self.feats].values)
		

	
class myMLP():
	def __init__(self,  X_sets, Y_sets, parameter_set, feats):
		self.X_test_training = X_sets[0][feats]
		self.Y_test_training = Y_sets[0]
		self.X_test = X_sets[1][feats]
		self.Y_test = Y_sets[1]
		
		self.X_validation_training = X_sets[2][feats]
		self.Y_validation_training = Y_sets[2]
		self.X_validation = X_sets[3][feats]
		self.Y_validation = Y_sets[3]
		
		self.rmse_list = []
		
		
		rmse = 1000
		dump("Hidden nodes parameters to validate:", (parameter_set[0],parameter_set[-1]))
		bar = Bar(len(parameter_set))
		for hidden_layer_units in parameter_set:
			
			new_model = MLPRegressor(hidden_layer_sizes=(hidden_layer_units,), activation='relu', max_iter=10000,warm_start=True).fit(self.X_validation_training.values, self.Y_validation_training.values.ravel())
			
			new_rmse = RMSE(new_model.predict(self.X_validation.values), self.Y_validation.values)
			
			self.rmse_list.append(new_rmse)
			
			if(new_rmse < rmse):
				rmse = new_rmse
				self.parameter = hidden_layer_units
			
			bar.moreBar()
			
		bar.noBar()
			
		dump("Hidden nodes:", self.parameter)

	def test(self):
		self.model = MLPRegressor(hidden_layer_sizes=(self.parameter,), activation='relu', max_iter=100000,warm_start=True).fit(self.X_test_training.values, self.Y_test_training.values.ravel())
		rmse = RMSE(self.model.predict(self.X_test.values), self.Y_test.values)
		return rmse

	def getHistory(self):
		return self.rmse_list
	