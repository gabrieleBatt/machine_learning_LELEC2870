import matplotlib.pyplot as plt
import matplotlib as mpl
import random
######################################################
from common import *
from feature_selection import *
from data_handler import *
from model_maker import *


######################################################
	
X1, Y1 = readData()

X1_normalized = normalizeFeatures(X1)

dump("Normalized:", X1_normalized.columns.values)

######################################################

X1_selected = selectFeaturesCorrelation(X1_normalized, Y1, 0.1)

dump("Selected - LM:", X1_selected.columns.values)

linear_model = makeLinearModel(X1_selected, Y1, 0.2)

######################################################

X1_selected = selectFeaturesMI(X1_normalized, Y1, 0.1)

dump("Selected - KNN:", X1_selected.columns.values)

k_set = [i for i in range(20,1000)]

KNN_model = makeKNN(X1_selected, Y1, k_set, 0.2, 0.3)

######################################################

X1_selected = selectFeaturesMI(X1_normalized, Y1, 0.1)

dump("Selected - RBFN:", X1_selected.columns.values)

#parameter_set = []
#for i in range(40,60):
	#for j in np.logspace(-2,2,num=10):
		#parameter_set.append((int(i),j))

parameter_set = [[43,35]]

RBFN_model = makeRBFN(X1_selected, Y1, parameter_set, 0.2, 0.3)

######################################################


