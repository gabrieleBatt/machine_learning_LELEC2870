import numpy as np
import pandas as pd

from common import *

def selectFeatures(Xn, Y, min_cor):
	n_feats = Xn.values.T.shape[0]
	priority_list = []
	for column in Xn.columns.values:
		correlation = np.corrcoef(Xn[column].values.T, Y.values.T[0])[0][1]
		priority_list.append((column,correlation))
		
	priority_list.sort(reverse = True)
	
	discarded_feats = [column for column,correlation in priority_list if np.abs(correlation) < min_cor]
	
	Xn = Xn.drop(columns=discarded_feats)
	return Xn
	
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
	