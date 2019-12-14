import math
import random
from sklearn.metrics import mean_squared_error

YEAR = 'year'
MONTH = 'month'  
DAY = 'day'    
NDATE = 'normDate'    
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
	print("############################")
	print(str, obj)

	
def RMSE(Y_actual, Y_predicted):
	return math.sqrt(mean_squared_error(Y_actual, Y_predicted))
	
	
def divideSets(X, Y, ratio):
		id_set = list(X.index)
		random.Random(0).shuffle(id_set)
		
		test_id_set = id_set[:int(X.shape[0]*ratio)]
		training_id_set = id_set[int(X.shape[0]*ratio):]

		X_training =  X.drop(test_id_set,axis=0)
		Y_training =  Y.drop(test_id_set,axis=0)

		X_test = X.drop(training_id_set,axis=0)
		Y_test = Y.drop(training_id_set,axis=0)
		return (X_training, Y_training, X_test, Y_test)