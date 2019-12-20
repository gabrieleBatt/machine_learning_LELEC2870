import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import ExtraTreesClassifier
from scipy import stats

from common import *

#Returns n_feats number of features, with either the higher correation or mutual information index, according to the method ('CORR' or 'MI') specified; returns also the one-hot encoding of the section
def selectFeatures(Xn, Y, n_feats, method, dummy_vars):
	Xn = Xn.drop(columns=dummy_vars)
	feature_list = []
	mutual_info_list = mutual_info_regression(Xn.values, Y.values.T[0])
	for column in Xn.columns.values:
		if(method=='MI'):
			value = mutual_info_regression(Xn[column].values.reshape(-1,1), Y.values.T[0])
		elif(method == 'CORR'):
			value = np.corrcoef(Xn[column].values.T, Y.values.T[0])[0][1]
		feature_list.append((column,np.abs(value)))
	
	feature_list.sort(key=lambda x: x[1], reverse=True)
	
	selected_features = []
	for feat in Xn.drop(columns=[column for column,_ in feature_list[n_feats:]]).columns.values:
		selected_features.append(feat)
	for feat in dummy_vars:
		selected_features.append(feat)
	return  selected_features

#Converts year, month, day, and hour in one variable: 
#hours from 1 jan 2013
def normalizeDate(year, month, day, hour):
	normalized = []
	for i in range(day.shape[0]):
		month_base = {
		1: 0,
		2: 31,
		3: 59,
		4: 90,
		5: 120,
		6: 151,
		7: 181,
		8: 212,
		9: 243,
		10: 273,
		11: 304,
		12: 334
		}
		days = day[i]+month_base.get(month[i], "Invalid month")+(year[i]-2013)*365.25
		hours = (days-1)*24 + hour[i]
		normalized.append(hours)	
	return normalized

#Wind direction converted into two trigonometrical variables
def normalizeWindDirection(wind_dir):
	sin = []
	cos = []
	for wd in wind_dir:
		sum = 0
		for c in wd:
			if (c == 'N'):
				sum += np.pi/2
			if (c == 'W'):
				sum += np.pi
			if (c == 'S'):
				sum += 3*np.pi/2
		angle = sum/len(wd)
		sin.append(np.sin(angle))
		cos.append(np.cos(angle))
	return (sin,cos)

#Removal of outlier data according to ther Z-score
def removeOutliers(X, Y):
	df = X.copy()
	df[PM25]=Y[PM25].values
	df = df.drop(columns=[STATION])
	z = np.abs(stats.zscore(df))
	outliers = set(np.where(z > 2)[0])
	dump("Removing outliers:", len(outliers)) 
	X = X.drop(outliers,axis=0)
	Y = Y.drop(outliers,axis=0)
	index = [i for i in range(X.shape[0])]
	X = X.reindex(index, method='backfill')
	Y = Y.reindex(index, method='backfill')
	return (X,Y)

#The normalized data contains the manipulated features as explained in section 1 of the report(Feature selection)	
def normalizeFeatures(X, Y):
	X = X.copy()
	Y = Y.copy()
	#Date converted in one variable
	X[NDATE] = normalizeDate(X[YEAR], X[MONTH], X[DAY], X[HOUR])
	X = X.drop(columns=[YEAR,MONTH,DAY,HOUR])
	
	#Wind direction converted into two trigonometrical variables
	X[WDS], X[WDC] = normalizeWindDirection(X[WD])
	X = X.drop(columns=[WD])
	
	#All columns distributions (except the station) are normalized
	for column in X.columns:
		if column != STATION:
			mean = np.mean(X[column].values)
			var = np.var(X[column].values)
			X[column] = [(value-mean)/np.sqrt(var) for value in X[column].values]
	
	#Removal of outlier data
	X,Y = removeOutliers(X,Y)
		
	#One-hot encoding of the station
	X[STATION] = pd.Categorical(X[STATION])
	dfONE = pd.get_dummies(X[STATION], prefix = STATION)
	X = X.drop(columns=[STATION])
	
	real_feats = X.columns.values
	X = pd.concat([X, dfONE], axis=1)
	
	#returns the normalized data frames of input and output, 
	#the names of the dummy variables(one-hot encoding of station, 
	#and the names of the other features(real_feats)
	return (X,Y, dfONE.columns.values, real_feats)
	