from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

from common import *
from rbfn import *

class myLinearModel():
	def __init__(self, X_selected, Y, test_dim, val_dim):
		self.X_final_training, self.Y_final_training, self.X_test, self.Y_test = divideSets(X_selected, Y, test_dim)
	
		self.X_training, self.Y_training, self.X_validation, self.Y_validation = divideSets(self.X_final_training, self.Y_final_training, val_dim)
	
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
	
	def getModel():
		return self.model
	
def makeKNN(X_selected, Y, k_set,  test_dim, val_dim):
	X_training, Y_training, X_test, Y_test = divideSets(X_selected, Y, test_dim)
	
	X_training, Y_training, X_validation, Y_validation = divideSets(X_training, Y_training, val_dim)
	
	rmse = 1000
	for n_n in k_set:
	
		new_model = KNeighborsRegressor(n_neighbors=n_n).fit(X_training.values, Y_training.values)
		
		new_rmse = RMSE(new_model.predict(X_validation.values), Y_validation.values)
		
		if(new_rmse < rmse):
			rmse = new_rmse
			k = n_n
			model = new_model
	
	dump("K:", k)
	
	dump("Score KNN Model:", model.score(X_test.values, Y_test.values))
	dump("RMSE KNN Model:", RMSE(model.predict(X_test.values), Y_test.values))
	
	

def makeRBFN(X_selected, Y, parameter_set,  test_dim, val_dim):
	
	X_training, Y_training, X_test, Y_test = divideSets(X_selected, Y, test_dim)
	
	X_training, Y_training, X_validation, Y_validation = divideSets(X_training, Y_training, val_dim)
	

	rmse = 1000
	for nb_centers, width_scaling in parameter_set:
	
		new_model = RBFN(nb_centers, width_scaling).fit(X_training.values, Y_training.values)
		
		new_rmse = RMSE(new_model.predict(X_validation.values), Y_validation.values)
		
		if(new_rmse < rmse):
			rmse = new_rmse
			parameters = (nb_centers,width_scaling)
			model = new_model
	
	dump("Parameters:", parameters)
	
	dump("Score RBFN Model:", model.score(X_test.values, Y_test.values))
	dump("RMSE RBFN Model:", RMSE(model.predict(X_test.values), Y_test.values))
	
	
def makeMLP(X_selected, Y, parameter_set,  test_dim, val_dim):
	
	X_training, Y_training, X_test, Y_test = divideSets(X_selected, Y, test_dim)
	
	X_training, Y_training, X_validation, Y_validation = divideSets(X_training, Y_training, val_dim)
	

	rmse = 1000
	for hidden_layer_units in parameter_set:
	
		new_model = MLPRegressor((hidden_layer_units,hidden_layer_units), activation='relu', max_iter=10000).fit(X_training.values, Y_training.values.ravel())
		
		new_rmse = RMSE(new_model.predict(X_validation.values), Y_validation.values)
		
		if(new_rmse < rmse):
			rmse = new_rmse
			parameter = hidden_layer_units
			model = new_model
		dump("RMSE Validation MLP-%d:"%hidden_layer_units, RMSE(model.predict(X_validation.values), Y_validation.values))
	
	dump("Hidden layer units:", parameter)

	dump("Score MLP:", model.score(X_test.values, Y_test.values))
	dump("RMSE MLP:", RMSE(model.predict(X_test.values), Y_test.values))
	
class myMLP():
	def __init__(self, X_selected, Y, parameter_set,  test_dim, val_dim):
		self.X_final_training, self.Y_final_training, self.X_test, self.Y_test = divideSets(X_selected, Y, test_dim)
	
		self.X_training, self.Y_training, self.X_validation, self.Y_validation = divideSets(self.X_final_training, self.Y_final_training, val_dim)

		rmse = 1000
		for hidden_layer_units in parameter_set:
		
			new_model = MLPRegressor((hidden_layer_units,hidden_layer_units), activation='relu', max_iter=10000).fit(self.X_training.values, self.Y_training.values.ravel())
			
			new_rmse = RMSE(new_model.predict(self.X_validation.values), self.Y_validation.values)
			
			if(new_rmse < rmse):
				rmse = new_rmse
				self.parameter = hidden_layer_units
				self.model = new_model

	def dumpScore(self):
		dump("Score MLP:", model.score(self.X_test.values, self.Y_test.values))
		return self
	
	def validate(self):
		rmse = RMSE(self.model.predict(self.X_validation.values), self.Y_validation.values)
		return rmse
	
	def test(self):
		self.model = MLPRegressor((self.parameter,self.parameter), activation='relu', max_iter=10000).fit(self.X_final_training.values, self.Y_final_training.values.ravel())
		rmse = RMSE(self.model.predict(self.X_test.values), self.Y_test.values)
		return rmse
		
	def getModel(self):
		return self.model
		
	def getParameter(self):
		return self.parameter
	