from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

from common import *
from rbfn import *

def makeLinearModel(X_selected, Y, test_dim):
	X_training, Y_training, X_test, Y_test = divideSets(X_selected, Y, test_dim)

	model = LinearRegression().fit(X_training.values, Y_training.values)

	dump("Score Linear Model:", model.score(X_test.values, Y_test.values))
	dump("RMSE Linear Model:", RMSE(model.predict(X_test.values), Y_test.values))
	
	return model
	
def makeKNN(X_selected, Y, k_set,  test_dim, val_dim):
	X_training, Y_training, X_test, Y_test = divideSets(X_selected, Y, test_dim)
	
	X_training, Y_training, X_validation, Y_validation = divideSets(X_training, Y_training, val_dim)
	
	rmse = 1000
	for n_n in k_set:
	
		new_model = KNeighborsRegressor(n_neighbors=n_n).fit(X_training.values, Y_training.values)
		
		new_rmse = RMSE(new_model.predict(X_test.values), Y_test.values)
		
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
		
		new_rmse = RMSE(new_model.predict(X_test.values), Y_test.values)
		
		if(new_rmse < rmse):
			rmse = new_rmse
			parameters = (nb_centers,width_scaling)
			model = new_model
	
	dump("Parameters:", parameters)
	
	dump("Score RBFN Model:", model.score(X_test.values, Y_test.values))
	dump("RMSE RBFN Model:", RMSE(model.predict(X_test.values), Y_test.values))
	
	