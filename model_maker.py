from sklearn.linear_model import LinearRegression

from common import *
from rbfn import *

def makeLinearModel(X_selected, Y, test_dim):
	X_training, Y_training, X_test, Y_test = divideSets(X_selected, Y, test_dim)

	model = LinearRegression().fit(X_training.values, Y_training.values)

	dump("Score Linear Model:", model.score(X_test.values, Y_test.values))
	dump("RMSE Linear Model:", RMSE(model.predict(X_test.values), Y_test.values))
	
	return model

def makeRBFN(X_selected, Y, parameter_set,  test_dim, val_dim):
	
	X_training, Y_training, X_test, Y_test = divideSets(X_selected, Y, test_dim)
	
	X_training, Y_training, X_validation, Y_validation = divideSets(X_training, Y_training, val_dim)
	
	parameters = (0,0)
	model = RBFN(0,0)
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
	
	