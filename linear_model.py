import numpy as np
from sklearn.linear_model import LinearRegression

from common import *


class LinearModel:
	
	def __init__(self, X_training_set, Y_training_set):
		self.model = LinearRegression().fit(X_training_set, Y_training_set)
		
	def test(self, X_test_set, Y_actual):
		Y_predicted = self.model.predict(X_test_set)
		
		return RMSE(Y_actual, Y_predicted)