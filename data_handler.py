import pandas as pd

from common import *

def readData():
	X1 = pd.read_csv("X1.csv")
	Y1 = pd.read_csv("Y1.csv",header=None,names=[PM25])
	return (X1, Y1)