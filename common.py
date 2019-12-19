import math
import random
from sklearn.metrics import mean_squared_error
import pandas as pd

YEAR = 'year'
MONTH = 'month'  
DAY = 'day'    
NDATE = 'DATE'    
HOUR = 'hour'
HS = 'hourSin'
HC = 'hourCos'
SO2 = 'SO2'
NO2 = 'NO2'
CO = 'CO'
O3 = 'O3'
TEMP = 'TEMP'
PRES = 'PRES'
DEWP = 'DEWP'
RAIN = 'RAIN'   
WD = 'wd'
WDS = 'wdSin'
WDC = 'wdCos'
WSPM = 'WSPM'
STATION = 'station'
PM25 = 'PM2.5'

def dump(str, obj):
	print("--------------------------------------------------------------------------------------")
	print(str, obj)

	
def RMSE(Y_actual, Y_predicted):
	return math.sqrt(mean_squared_error(Y_actual, Y_predicted))
	
	
def divideSets(X, Y):
		small_id_set = [i for i in range(0,len(list(X.index)),5)]
		big_id_set = [i for i in range(0,len(list(X.index))) if i not in small_id_set]

		index = [i for i in range(len(small_id_set))]
		X_s =  X.drop(small_id_set,axis=0).reindex(index, method='backfill')
		Y_s =  Y.drop(small_id_set,axis=0).reindex(index, method='backfill')

		
		index = [i for i in range(len(big_id_set))]
		X_b = X.drop(big_id_set,axis=0).reindex(index, method='backfill')
		Y_b = Y.drop(big_id_set,axis=0).reindex(index, method='backfill')
		return (X_b, Y_b, X_s, Y_s)
		
def readData():
	X1 = pd.read_csv("X1.csv")
	Y1 = pd.read_csv("Y1.csv",header=None,names=[PM25])
	return (X1, Y1)