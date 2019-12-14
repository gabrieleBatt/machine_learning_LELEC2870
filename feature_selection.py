import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

from common import *

def discardFeatures(Xn, feature_list, min_value):
	discarded_feats = [column for column,value in feature_list if value < min_value]
	
	Xn = Xn.drop(columns=discarded_feats)
	return Xn

def selectFeaturesMI(Xn, Y, min_value):
	feature_list = []
	mutual_info_list = mutual_info_regression(Xn.values, Y.values.T[0])
	for column in Xn.columns.values:
		mutual_info = mutual_info_regression(Xn[column].values.reshape(-1,1), Y.values.T[0])
		feature_list.append((column,mutual_info))
	
	return discardFeatures(Xn, feature_list, min_value)
	
def selectFeaturesCorrelation(Xn, Y, min_value):
	feature_list = []
	for column in Xn.columns.values:
		correlation = np.corrcoef(Xn[column].values.T, Y.values.T[0])[0][1]
		feature_list.append((column,np.abs(correlation)))
		
	feature_list.sort(reverse = True)
	
	return discardFeatures(Xn, feature_list, min_value)
	
def normalizeDate(year, day, month):
	normalized = []
	for i in range(day.shape[0]):
		base = {
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
		days = day[i]+base.get(month[i], "Invalid month")
		if(year[i] == 2016 and month[i] > 2):
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
		angle = 2*np.pi*(h-1)/12
		sin.append(np.sin(angle))
		cos.append(np.cos(angle))
	return (sin,cos)
	
def normalizeFeatures(X):
	X = X.drop(columns=STATION)
	
	X[NDATE] = normalizeDate(X[YEAR], X[DAY], X[MONTH])
	X = X.drop(columns=[MONTH,DAY])
	
	X[WDS], X[WDC] = normalizeWindDirection(X[WD])
	X = X.drop(columns=[WD])
	
	X[HS], X[HC] = normalizeHour(X[HOUR])
	X = X.drop(columns=[HOUR])
	
	return X
	