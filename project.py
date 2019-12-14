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

#X1_selected = selectFeaturesCorrelation(X1_normalized, Y1, 0.2)

#dump("Selected - LM:", X1_selected.columns.values)

#linear_model = makeLinearModel(X1_selected, Y1, 0.2, 0.3)

######################################################

X1_selected = selectFeaturesMI(X1_normalized, Y1, 0.7)

dump("Selected - NLM:", X1_selected.columns.values)

parameter_set = []
for i in range(10):
	for j in range(10):
		parameter_set.append((i,j/20.0))

linear_model = makeRBFN(X1_selected, Y1, parameter_set, 0.2, 0.3)

######################################################


