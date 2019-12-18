import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import random
import pandas as pd
######################################################
from common import *
from feature_selection import *
from data_handler import *
from model_maker import *
from sklearn.feature_selection import mutual_info_regression


######################################################

options = sys.argv[1:]
dump("Options: ", options)

do_all = ('-all' in options)

######################################################

X1, Y1 = readData()

X1_normalized, Y1_normalized = normalizeFeatures(X1, Y1)

n_station = len(set(X1_normalized[STATION].values))

X1_per_station_set = [pd.DataFrame(columns=X1_normalized.columns.values) for _ in range(len(X1_normalized.columns))]
Y1_per_station_set = [pd.DataFrame(columns=[PM25]) for _ in range(len(X1_normalized.columns))]

for i,station in enumerate(X1_normalized[STATION].values):
	X1_per_station_set[station].loc[i] = X1_normalized.values[i]
	Y1_per_station_set[station].loc[i] = Y1_normalized.values[i]

for i in range(n_station):
	index = [i for i in range(len(X1_per_station_set[i]))]
	X1_per_station_set[i] = X1_per_station_set[i].reindex(index, method='backfill')
	Y1_per_station_set[i] = Y1_per_station_set[i].reindex(index, method='backfill')
	
X1_sets = []
Y1_sets = []
for i in range(n_station):
	X1_test_training, Y1_test_training, X1_test, Y1_test = divideSets(X1_per_station_set[i], Y1_per_station_set[i])
	
	X1_validation_training, Y1_validation_training, X1_validation, Y1_validation = divideSets(X1_test_training, Y1_test_training)

	X1_sets.append((X1_test_training,X1_test,X1_validation_training, X1_validation))
	Y1_sets.append((Y1_test_training,Y1_test,Y1_validation_training, Y1_validation))		

######################################################
######################################################

if(do_all or '-lm' in options or '-m' in options):
	
	model_per_station = []
	for station in range(n_station):
		feats_selected = selectFeatures(X1_per_station_set[station], Y1_per_station_set[station], 4, 'CORR').columns.values

		model = myLinearModel(X1_sets[station][0], Y1_sets[station][0], feats_selected)
		
		model_per_station.append(model)
		
		
	predictions = []
	actual = []
	for station in range(n_station):
		for element in model_per_station[station].predict(X1_sets[station][1]):
			predictions.append(element[0])
		for element in Y1_sets[station][1].values:
			actual.append(element[0])
	
	dump("RMSE test Linear Model:", RMSE(actual,predictions))


######################################################

if(do_all or '-mlp' in options or '-m' in options):

	parameter_set = [i for i in range(5,30)]
	
	model_per_station = []
	for station in range(n_station):
		feats_selected = selectFeatures(X1_per_station_set[station], Y1_per_station_set[station], 4, 'MI').columns.values

		model = myLinearModel(X1_sets[station][0], Y1_sets[station][0], feats_selected)
		
		model_per_station.append(model)
		
		
	predictions = []
	actual = []
	for station in range(n_station):
		for element in model_per_station[station].predict(X1_sets[station][1]):
			predictions.append(element[0])
		for element in Y1_sets[station][1].values:
			actual.append(element[0])
	
	dump("RMSE MLP:", RMSE(actual,predictions))

######################################################

if(do_all or '-knn' in options or '-m' in options):
	feats_selected = selectFeatures(X1_normalized, Y1_normalized, 4, 'MI').columns.values

	dump("Selected - KNN:", feats_selected)

	k_set = [i for i in range(10,100)]
	
	KNN_model = myKNN(X1_sets, Y1_sets, k_set, feats_selected)
	rmse = KNN_model.test()
	dump("K:", KNN_model.getK())
	dump("RMSE KNN Model:", rmse)
	
######################################################

if(do_all or '-data' in options or '-avg' in options or '-var' in options or '-plot' in options):
	plt.figure(0)
	n_feats = len(X1_normalized.columns)
	y = Y1_normalized.values
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
		corr = corr + [np.abs(np.corrcoef(X1_normalized[column].values.T, Y1_normalized.values.T[0])[0][1])]
	mis = []
	for column in X1_normalized.columns.values:
		mis = mis + [np.abs(mutual_info_regression(X1_normalized[column].values.reshape(-1,1), Y1_normalized.values.T[0])[0])]
	
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