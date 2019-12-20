import sys
import math
import pickle
import pandas as pd

from sklearn.metrics import mean_squared_error

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
	X2 = pd.read_csv("X2.csv")
	Y1 = pd.read_csv("Y1.csv",header=None,names=[PM25])
	return (X1, Y1, X2)
	
def writeData(Y):
	df = pd.DataFrame(Y)
	df.to_csv(path_or_buf='Y2.csv',index=False, header=False)

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

#To have consistent runs, the training, validation and testing sets are fixed and sved in files		
with open('train_test.data', 'rb') as filehandle:
    TEST_TRAINING_INDECES = pickle.load(filehandle)
with open('test.data', 'rb') as filehandle:
    TEST_INDECES = pickle.load(filehandle)
with open('train_validation.data', 'rb') as filehandle:
    VALIDATION_TRAINING_INDECES = pickle.load(filehandle)
with open('validation.data', 'rb') as filehandle:
    VALIDATION_INDECES = pickle.load(filehandle)