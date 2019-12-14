import matplotlib.pyplot as plt
import matplotlib as mpl
import random
######################################################
from common import *
from feature_selection import *
from data_handler import *
from linear_model import *


######################################################
	
X1, Y1 = readData()

X1_normalized = normalizeFeatures(X1)

dump("Normalized:", X1_normalized.columns.values)

######################################################

X1_selected = selectFeaturesCorrelation(X1_normalized, Y1)

dump("Selected - LM:", X1_selected.columns.values)

LinearModel(X1_selected, Y1, 0.2)

######################################################


