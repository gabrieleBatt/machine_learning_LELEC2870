import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA

from common import *

def selectFeatures(Xn, Y, n_feats, method):
	feature_list = []
	mutual_info_list = mutual_info_regression(Xn.values, Y.values.T[0])
	for column in Xn.columns.values:
		if(method=='MI'):
			value = mutual_info_regression(Xn[column].values.reshape(-1,1), Y.values.T[0])
		elif(method == 'CORR'):
			value = np.corrcoef(Xn[column].values.T, Y.values.T[0])[0][1]
		feature_list.append((column,value))
	
	feature_list.sort()
	
	return Xn.drop(columns=[column for column,_ in feature_list[n_feats:]])
	

def normalizeDate(year, day, month):
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
		days = day[i]+month_base.get(month[i], "Invalid month")+(year[i]-2013)*365
		if(year[i] > 2016 or (year[i] == 2016 and month[i] > 2)):
			days += 1
		normalized.append(days)	
	return normalized
	
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
	
def normalizeHour(hours):
	sin = []
	cos = []
	for h in hours:
		angle = 2*np.pi*h/24
		sin.append(np.sin(angle))
		cos.append(np.cos(angle))
	return (sin,cos)
	
def normalizeFeatures(X):
	X = X.drop(columns=STATION)
	
	X[NDATE] = normalizeDate(X[YEAR], X[DAY], X[MONTH])
	X = X.drop(columns=[YEAR,MONTH,DAY])
	
	X[WDS], X[WDC] = normalizeWindDirection(X[WD])
	X = X.drop(columns=[WD])
	
	X[HS], X[HC] = normalizeHour(X[HOUR])
	X = X.drop(columns=[HOUR])
	
	#for column in X.columns:
	#	mean = np.mean(X[column].values)
	#	var = np.var(X[column].values)
	#	X[column] = [(value-mean)/np.sqrt(var) for value in X[column].values]
		
	return X
	