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
from sklearn.feature_selection import mutual_info_regression


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

if(do_all or '-lm' in options or '-m' in options):
	X1_selected = selectFeatures(X1_normalized, Y1, 4, 'CORR')

	dump("Selected - LM:", X1_selected.columns.values)

	rmse = myLinearModel(X1_selected, Y1, 0.2, 0).test()
	dump("RMSE test Linear Model:", rmse)


######################################################

if(do_all or '-wlm' in options or '-wm' in options):
	
	features_queque = selectFeatures(X1_normalized, Y1, len(X1_normalized.columns), 'CORR')
	
	feats_selected = []
	rmse_min = 1000
	while(len(feats_selected) <= 5):
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
	
	dump("Selected - WLM:", feats_selected)
	rmse = myLinearModel(X1_normalized[feats_selected], Y1, 0.1, 0.2).test()
	dump("RMSE test Linear Model:", rmse)

######################################################

if(do_all or '-mlp' in options or '-m' in options):

	parameter_set = [i for i in range(7,8)]
	
	X1_selected = selectFeatures(X1_normalized, Y1, 5, 'MI')
	
	dump("Selected - MLP:", X1_selected.columns.values)	
	MLP_model = myMLP(X1_selected, Y1, parameter_set, 0.2, 0.2)
	rmse = MLP_model.test()
	dump("Parameter: ", MLP_model.getParameter())
	dump("MLP test Linear Model:", rmse)

######################################################

if(do_all or '-wmlp' in options or '-m' in options):

	parameter_set = [i for i in range(14,15)]
	
	features_queque = selectFeatures(X1_normalized, Y1, len(X1_normalized.columns), 'MI')
	
	feats_selected = []
	rmse_min = 1000
	while(len(feats_selected) <= 5):
		new_feat = None
		for feat in features_queque.columns:
			feats_temp = feats_selected + [feat]
			rmse = myMLP(X1_normalized[feats_temp], Y1, parameter_set, 0.2, 0.2).validate()
			if(rmse < rmse_min):
				rmse_min = rmse
				new_feat = feat
		if new_feat is not None:
			feats_selected = feats_selected + [new_feat]
			features_queque = features_queque.drop(columns=[new_feat])
		else: 
			break
	
	
	dump("Selected - WMLP:", feats_selected)	
	MLP_model = myMLP(X1_normalized[feats_selected], Y1, parameter_set, 0.2, 0.2)
	rmse = MLP_model.test()
	dump("Parameter: ", MLP_model.getParameter())
	dump("MLP test Linear Model:", rmse)

######################################################

if(do_all or '-knn' in options or '-m' in options):
	X1_selected = selectFeatures(X1_normalized, Y1, 4, 'MI')

	dump("Selected - KNN:", X1_selected.columns.values)

	k_set = [i for i in range(10,100)]
	
	KNN_model = myKNN(X1_selected, Y1, k_set, 0.2, 0.2)
	rmse = KNN_model.test()
	dump("K:", KNN_model.getK())
	dump("RMSE KNN Model:", rmse)
	
######################################################

if(do_all or '-data' in options or '-avg' in options or '-var' in options or '-plot' in options):
	plt.figure(0)
	n_feats = len(X1_normalized.columns)
	y = Y1.values
	for i,column in enumerate(X1_normalized):
		x = X1_normalized[column].values
		plt.subplot(1+(n_feats/4), 4, i+1)
		if('-data' in options or '-plot' in options):
			plt.scatter(x, y, alpha=0.5)
		if('-avg' in options or '-var' in options or '-plot' in options):
			sorted_values = [(x[i], y[i]) for i in range(len(x))]
			sorted_values.sort()
			avg = []
			var = []
			step = (sorted_values[-1][0]-sorted_values[0][0])/20
			i = 0
			while i < len(sorted_values):
				sum = []
				j = i
				while j < len(sorted_values) and np.abs(sorted_values[j][0]-sorted_values[i][0]) < step:
					sum = sum + [sorted_values[j][1]]
					j += 1
				avg = avg + [(sorted_values[i][0], np.mean(sum))]
				var = var + [(sorted_values[i][0], np.sqrt(np.var(sum)))]
				i = j+1
			if('-avg' in options or '-plot' in options):
				plt.plot([v[0] for v in avg], [v[1] for v in avg], 'r--')
			if('-var' in options or '-plot' in options):
				plt.plot([v[0] for v in var], [v[1] for v in var], 'g--')
		plt.title(column)
		plt.ylabel(PM25)
		frame1 = plt.gca()
		frame1.axes.xaxis.set_ticklabels([])

######################################################

if (do_all or '-filter' in options or '-f' in options):
	names = X1_normalized.columns.values
	corr = []
	for column in X1_normalized.columns.values:
		corr = corr + [np.abs(np.corrcoef(X1_normalized[column].values.T, Y1.values.T[0])[0][1])]
	mis = []
	for column in X1_normalized.columns.values:
		mis = mis + [np.abs(mutual_info_regression(X1_normalized[column].values.reshape(-1,1), Y1.values.T[0])[0])]
	
	plt.figure(1)
	plt.subplot(121)
	plt.bar(names, corr)
	plt.title("Correlation")
	plt.subplot(122)
	plt.bar(names, mis)
	plt.title("Mututal Information")

######################################################

if(do_all or '-data' in options or '-avg' in options or '-var' in options or '-plot' in options or '-filter' in options or '-f' in options):
	plt.show()