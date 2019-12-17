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
	X1_selected = selectFeatures(X1_normalized, Y1, 5, 'CORR')

	dump("Selected - LM:", X1_selected.columns.values)

	rmse = myLinearModel(X1_selected, Y1, 0.2, 0).test()
	dump("RMSE test Linear Model:", rmse)

######################################################

if(do_all or '-knn' in options or '-m' in options):
	X1_selected = selectFeatures(X1_normalized, Y1, 4, 'MI')

	dump("Selected - KNN:", X1_selected.columns.values)

	k_set = [i for i in range(20,50)]

	KNN_model = makeKNN(X1_selected, Y1, k_set, 0.2, 0.3)

######################################################

if(do_all or '-rbfn' in options or '-m' in options):

	X1_selected = selectFeatures(X1_normalized, Y1, 14, 'MI')

	dump("Selected - RBFN:", X1_selected.columns.values)

	#parameter_set = []
	#for i in range(40,60):
		#for j in np.logspace(-2,2,num=10):
			#parameter_set.append((int(i),j))

	parameter_set = [[43,35]]

	RBFN_model = makeRBFN(X1_selected, Y1, parameter_set, 0.2, 0.3)

######################################################

if(do_all or '-wmlp' in options or '-m' in options):

	parameter_set = [i for i in range(10,20)]
	
	features_queque = selectFeatures(X1_normalized, Y1, len(X1_normalized.columns), 'MI')
	
	feats_selected = []
	rmse_min = 1000
	while(len(features_queque) != 0):
		new_feat = None
		for feat in features_queque:
			feats_temp = feats_selected + [feat]
			rmse = myMLP(X1_normalized[feats_temp], Y1, parameter_set, 0.2, 0.2).validate()
			if(rmse < rmse_min):
				rmse_min = rmse
				new_feat = feat
		if new_feat is not None:
			feats_selected = feats_selected + [new_feat]
			features_queque.drop(columns=[new_feat])
		else: 
			break
	
	
	dump("Final selection - WMLP:", feats_selected)	
	MLP_model = myMLP(X1_normalized[feats_selected], Y1, parameter_set, 0.2, 0.2)
	rmse = MLP_model.test()
	dump("Parameter: ", MLP_model.getParameter())
	dump("MLP test Linear Model:", rmse)

######################################################

if(do_all or '-wlm' in options or '-wm' in options):
	
	features_queque = selectFeatures(X1_normalized, Y1, len(X1_normalized.columns), 'CORR')
	
	feats_selected = []
	rmse_min = 1000
	while(len(features_queque) != 0):
		new_feat = None
		for feat in features_queque:
			feats_temp = feats_selected + [feat]
			rmse = myLinearModel(X1_normalized[feats_temp], Y1, 0.2, 0.2).validate()
			if(rmse < rmse_min):
				rmse_min = rmse
				new_feat = feat
		if new_feat is not None:
			feats_selected = feats_selected + [new_feat]
			features_queque.drop(columns=[new_feat])
		else: 
			break
	
	dump("Final selection - WLM:", feats_selected)
	rmse = myLinearModel(X1_normalized[feats_selected], Y1, 0.1, 0.2).test()
	dump("RMSE test Linear Model:", rmse)

######################################################
