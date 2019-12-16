import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import random
######################################################
from common import *
from feature_selection import *
from data_handler import *
from model_maker import *


######################################################

do_all = (len(sys.argv) == 1)
options = sys.argv[1:]
if(not do_all):
	dump("Options: ", options)

######################################################

X1, Y1 = readData()

X1_normalized = normalizeFeatures(X1)

dump("Normalized:", X1_normalized.columns.values)

######################################################

######################################################

if(do_all or '-vf' in options):
	n_feats = len(X1_normalized.columns)
	y = Y1.values
	for i,column in enumerate(X1_normalized):
		x = X1_normalized[column].values
		#plt.figure(i+1)
		plt.subplot(1+(n_feats/4), 4, i+1)
		plt.scatter(x, y, alpha=0.5)
		plt.xlabel(column)
		plt.ylabel(PM25)
	plt.show()
######################################################

if(do_all or '-lm' in options or '-m' in options):
	X1_selected = selectFeatures(X1_normalized, Y1, 4, 'CORR')

	dump("Selected - LM:", X1_selected.columns.values)

	linear_model = makeLinearModel(X1_selected, Y1, 0.2)

######################################################

if(do_all or '-knn' in options or '-m' in options):
	X1_selected = selectFeatures(X1_normalized, Y1, 4, 'MI')

	dump("Selected - KNN:", X1_selected.columns.values)

	k_set = [i for i in range(20,50)]

	KNN_model = makeKNN(X1_selected, Y1, k_set, 0.2, 0.3)

######################################################

if(do_all or '-rbfn' in options or '-m' in options):

	X1_selected = selectFeatures(X1_normalized, Y1, 4, 'MI')

	dump("Selected - RBFN:", X1_selected.columns.values)

	#parameter_set = []
	#for i in range(40,60):
		#for j in np.logspace(-2,2,num=10):
			#parameter_set.append((int(i),j))

	parameter_set = [[43,35]]

	RBFN_model = makeRBFN(X1_selected, Y1, parameter_set, 0.2, 0.3)

######################################################

if(do_all or '-mlp' in options or '-m' in options):

	X1_selected = selectFeatures(X1_normalized, Y1, 4, 'MI')

	dump("Selected - MLP:", X1_selected.columns.values)

	parameter_set = [14]

	MLP_model = makeMLP(X1_selected, Y1, parameter_set, 0.1, 0.1)

######################################################

if(do_all or '-wlm' in options or '-wm' in options):
	
	features_queque = selectFeatures(X1_normalized, Y1, len(X1_normalized.columns), 'CORR')
	
	X1_selected = []
	
	while(len(features_queque) != 0):
	
		for feat in features_queque:
			
			dump("Selected - LM:", X1_selected.columns.values)

			linear_model = makeLinearModel(X1_selected, Y1, 0.2)

######################################################
