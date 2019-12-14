import random
import numpy as np
from sklearn.linear_model import LinearRegression

from common import *

class LinearModel:
	
	def __init__(self, X_selected, Y, test_dim):
		X_training, Y_training, X_test, Y_test = self.createSets(X_selected, Y, test_dim)

		self.model = LinearRegression().fit(X_training, Y_training)

		dump("Score Linear Model:", self.model.score(X_test.values, Y_test.values))
		dump("RMSE Linear Model:", RMSE(self.model.predict(X_test.values), Y_test))
		
	def createSets(self, X_selected, Y, test_dim):
		id_set = list(range(X_selected.shape[0]))
		random.Random(0).shuffle(id_set)
		
		test_id_set = id_set[:int(X_selected.shape[0]*test_dim)]
		training_id_set = id_set[int(X_selected.shape[0]*test_dim):]

		X_training =  X_selected.drop(test_id_set,axis=0)
		Y_training =  Y.drop(test_id_set,axis=0)

		X_test = X_selected.drop(training_id_set,axis=0)
		Y_test = Y.drop(training_id_set,axis=0)
		return (X_training, Y_training, X_test, Y_test)