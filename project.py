import matplotlib.pyplot as plt
import matplotlib as mpl
import random
######################################################
from common import *
from feature_selection import *
from data_handler import *
from linear_model import *

TEST_DIM = 0.2

######################################################
	
X1, Y1 = readData()

X1_normalized = normalizeFeatures(X1)

dump("Normalized:", X1_normalized.columns.values)

X1_selected = selectFeatures(X1_normalized, Y1, 0.7)

dump("Selected:", X1_selected.columns.values)

######################################################

id_set = list(range(X1_selected.shape[0]))
random.Random(0).shuffle(id_set)
test_id_set = id_set[:int(X1_selected.shape[0]*TEST_DIM)]
training_id_set = id_set[int(X1_selected.shape[0]*TEST_DIM):]

X1_training =  X1_selected.drop(test_id_set,axis=0)
Y1_training =  Y1.drop(test_id_set,axis=0)

X1_test = X1_selected.drop(training_id_set,axis=0)
Y1_test = Y1.drop(training_id_set,axis=0)

######################################################

linear_model = LinearModel(X1_training.values, Y1_training.values)

dump("RMSE Linear Model:", linear_model.test(X1_test.values, Y1_test.values))

######################################################


