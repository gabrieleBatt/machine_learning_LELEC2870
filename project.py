import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import random
######################################################
from common import *
from feature_selection import *
from model_maker import *
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split
######################################################

options = sys.argv[1:]
dump("Options: ", options)
#If [-all] is a command line argument, all blocks will be executed
do_all = ('-all' in options)

######################################################

X1, Y1 = readData()

#The normalized data contains the manipulated features as explained in section 1 of the report(Feature selection)
#returns the normalized data frames of input and output, 
#the names of the dummy variables(one-hot encoding of station, 
#and the names of the other features(real_feats)
X1_normalized, Y1_normalized, dummy_vars, real_vars = normalizeFeatures(X1, Y1)

dump("Feats:", real_vars)
dump("Dummies:", dummy_vars)

#Here we divide the data in sets used for testing, training, and validation
df = X1_normalized.copy()
df[PM25] = Y1_normalized.values
train_test , test = train_test_split(df, test_size = 0.1)
train_validation , validation = train_test_split(train_test, test_size = 0.1)

X1_test_training = train_test.drop(columns=[PM25])
Y1_test_training = train_test[PM25]
X1_test = test.drop(columns=[PM25])
Y1_test = test[PM25]
X1_validation_training = train_validation.drop(columns=[PM25])
Y1_validation_training = train_validation[PM25]
X1_validation = validation.drop(columns=[PM25])
Y1_validation = validation[PM25]


X1_sets = (X1_test_training, X1_test, X1_validation_training, X1_validation)
Y1_sets = (Y1_test_training, Y1_test, Y1_validation_training, Y1_validation)

######################################################
#The following code is divided in blocks
#Each block is activated by command line argument
######################################################

#Activates on [-lm] and [-m]
#Uses a training set and test set
#Trains a linear regression model and tests it
if(do_all or '-lm' in options or '-m' in options):
	feats_selected = selectFeatures(X1_normalized, Y1_normalized, len(real_vars), 'CORR',dummy_vars)

	dump("Selected - LM:", feats_selected)

	rmse = myLinearModel(X1_sets, Y1_sets, feats_selected).test()
	dump("RMSE test Linear Model:", rmse)


######################################################

#Activates on [-knn] and [-m]
#Uses a training set and a test set
#The training set is divided in two to allow validation
#Trains a KNN model with Euclidean distance as distance function and validates for every parameter k in k_set
#After having selected the best k, it trains it again and tests it
if(do_all or '-knn' in options or '-m' in options):
	feats_selected = selectFeatures(X1_normalized, Y1_normalized, 6, 'CORR', dummy_vars)

	dump("Selected - KNN:", feats_selected)

	k_set = [i for i in range(1,200)]
	
	KNN_model = myKNN(X1_sets, Y1_sets, k_set, feats_selected)
	rmse = KNN_model.test()
	dump("RMSE KNN Model:", rmse)
	
	rmse_list = KNN_model.getHistory()
	plt.figure(21)
	plt.plot(k_set,rmse_list)
	
######################################################

#Activates on [-mlp] and [-m]
#Uses a training set and a test set
#The training set is divided in two to allow validation
#Trains a MLP model for every number of nodes in the hidden layer in parameter_set
#After having selected the best number of nodes in the hidden, it trains it again and tests it
if(do_all or '-mlp' in options or '-m' in options):

	parameter_set = [i for i in range(1,300)]
	
	feats_selected = selectFeatures(X1_normalized, Y1_normalized, 4, 'MI', dummy_vars)
	
	dump("Selected - MLP:", feats_selected)	
	MLP_model = myMLP(X1_sets, Y1_sets, parameter_set, feats_selected)
	rmse = MLP_model.test()
	dump("MLP test Linear Model:", rmse)
	
	rmse_list = MLP_model.getHistory()
	plt.figure(22)
	plt.plot(parameter_set,rmse_list)
	
######################################################

#Activates on [-data], [-avg], and [-plot]
#Uses the standard features 
#[-data] plots all the data; one graph per feature
#[-avg] plots a rough average: in each point the average is computed with the data closest to it; one graph per feature(ignores wind direction)
#If both [-data] and [-avg], or [-plot] are used, the plots will be overlapping
if(do_all or '-data' in options or '-avg' in options or '-plot' in options):
	plt.figure(10)
	n_feats = len(X1.columns)
	y = Y1.values
	for i,column in enumerate(X1.drop(columns=[STATION])):
		x = X1[column].values
		plt.subplot(1+(n_feats/4), 4, i+1)
		if('-data' in options or '-plot' in options):
			plt.scatter(x, y, alpha=0.5)
		if(('-avg' in options or '-plot' in options) and column != WD):
			sorted_values = [(x[i], y[i]) for i in range(len(x))]
			sorted_values.sort()
			avg = []
			step = (sorted_values[-1][0]-sorted_values[0][0])/20
			i = 0
			while i < len(sorted_values):
				sum = []
				j = i
				while j < len(sorted_values) and np.abs(sorted_values[j][0]-sorted_values[i][0]) < step:
					sum = sum + [sorted_values[j][1]]
					j += 1
				avg = avg + [(sorted_values[i][0], np.mean(sum))]
				i = j+1
			plt.plot([v[0] for v in avg], [v[1] for v in avg], 'r--')
		plt.title(column)
		plt.ylabel(PM25)
		frame1 = plt.gca()
		frame1.axes.xaxis.set_ticklabels([])
		
######################################################

#Activates on [-ndata], [-navg], and [-plot]
#Uses the features after manipulation and normalization
#[-ndata] plots all the data; one graph per feature
#[-navg] plots a rough average: in each point the average is computed with the data closest to it; one graph per feature
#If both [-ndata] and [-navg], or [-plot] are used, the plots will be overlapping
if(do_all or '-ndata' in options or '-navg' in options or '-plot' in options):
	plt.figure(11)
	n_feats = len(real_vars)
	y = Y1_normalized.values
	for i,column in enumerate(X1_normalized[real_vars]):
		x = X1_normalized[column].values
		plt.subplot(1+(n_feats/4), 4, i+1)
		if('-ndata' in options or '-plot' in options):
			plt.scatter(x, y, alpha=0.5)
		if('-navg' in options or '-plot' in options):
			sorted_values = [(x[i], y[i]) for i in range(len(x))]
			sorted_values.sort()
			avg = []
			step = (sorted_values[-1][0]-sorted_values[0][0])/20
			i = 0
			while i < len(sorted_values):
				sum = []
				j = i
				while j < len(sorted_values) and np.abs(sorted_values[j][0]-sorted_values[i][0]) < step:
					sum = sum + [sorted_values[j][1]]
					j += 1
				avg = avg + [(sorted_values[i][0], np.mean(sum))]
				i = j+1
			plt.plot([v[0] for v in avg], [v[1] for v in avg], 'r--')
		plt.title(column)
		plt.ylabel(PM25)
		frame1 = plt.gca()
		frame1.axes.xaxis.set_ticklabels([])

######################################################

#Activates on [-filter] and [-f]
#Uses the standard features
#It plots the correlation index and the mutual information index on a bar graph
if (do_all or '-filter' in options or '-f' in options):
	names = X1.drop(columns=[STATION,WD]).columns.values
	corr = []
	for column in names:
		corr = corr + [np.abs(np.corrcoef(X1[column].values.T, Y1.values.T[0])[0][1])]
	mis = []
	for column in names:
		mis = mis + [np.abs(mutual_info_regression(X1[column].values.reshape(-1,1), Y1.values.T[0])[0])]
	
	plt.figure(12)
	plt.subplot(121)
	plt.bar(names, corr)
	plt.title("Correlation")
	plt.subplot(122)
	plt.bar(names, mis)
	plt.title("Mututal Information")

######################################################

#Activates on [-nfilter] and [-nf]
#Uses the features after manipulation and normalization
#It plots the correlation index and the mutual information index on a bar graph
if (do_all or '-nfilter' in options or '-nf' in options):
	names = real_vars
	corr = []
	for column in real_vars:
		corr = corr + [np.abs(np.corrcoef(X1_normalized[column].values.T, Y1_normalized.values.T[0])[0][1])]
	mis = []
	for column in real_vars:
		mis = mis + [np.abs(mutual_info_regression(X1_normalized[column].values.reshape(-1,1), Y1_normalized.values.T[0])[0])]
	
	plt.figure(13)
	plt.subplot(121)
	plt.bar(names, corr)
	plt.title("Correlation")
	plt.subplot(122)
	plt.bar(names, mis)
	plt.title("Mututal Information")

######################################################

plt.show()