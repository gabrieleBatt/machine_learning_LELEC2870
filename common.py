import math
import random
import time
import sys
from sklearn.metrics import mean_squared_error
import pandas as pd

YEAR = 'year'
MONTH = 'month'  
DAY = 'day'
DS = 'daySin'    
DC = 'dayCos'    
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


def readData():
	X1 = pd.read_csv("X1.csv")
	Y1 = pd.read_csv("Y1.csv",header=None,names=[PM25])
	return (X1, Y1)

#Loading bar	
#Used to check status of very long computations
class Bar():
	
	def __init__(self, max_len):
		self.ratio = int(max_len/100)+1
		sys.stdout.write('|'+'-'*(int(max_len/self.ratio))+'|')
		self.len = 0
		self.moreBar()
		return 
	
	def moreBar(self):
		self.len += 1
		sys.stdout.write("\r"+'|'+'='*int(self.len/self.ratio))
		sys.stdout.flush()
	
	def noBar(self):
		sys.stdout.write('\r'+' '*(int(self.len/self.ratio)+2)+'\r')
		sys.stdout.flush()
		