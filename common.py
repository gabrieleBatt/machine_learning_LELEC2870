import math
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