import matplotlib.pyplot as plt
import numpy as np
import math
import csv

def logistic_function(x):
    rows = len(x) 
    num = np.ndarray(rows)
    denom = np.ndarray(rows)
    for i in range(rows):
        num[i] = 1;
        denom[i] = 1 + math.exp(-(-7.8923+2.262*math.log2(x[i]))) 
        # add 1.e-08 as a guard
    return (num / denom)

scaling_factor = 14 # Make each bubble size slightliy larger and more legible

reader = csv.reader(open('../data/logistic/BlowBF.csv','r'), delimiter=',')
result = list(reader)
rows = len(result)

raw_x = np.ndarray(rows, np.float64)
x = np.ndarray(rows, np.float64) # log2(diameter)
y = np.ndarray(rows, np.float64) # died/m
pop = np.ndarray(rows, np.float64) # Population

for i in range(rows):
    if (i != 0):
        raw_x[i] = float( result[i][1] )
        x[i] = math.log2( float(result[i][1]) )
        y[i] = float(result[i][2])/float(result[i][3])
        pop[i] = float(result[i][3])

x = x[1:]; y = y[1:]; raw_x = raw_x[1:]; pop = pop[1:]
# TODO: Use logarithmic plot
plt.title('Plot of The Blowdown Trees')
plt.xlabel('log$_2$(Diameter) [cm]')
plt.ylabel('Observed Blowdown Fraction [tumbled/total]')
plt.scatter(x=x, y=y, s=scaling_factor*pop, alpha=0.6, color='blue')
plt.plot(x, logistic_function(raw_x), color='red')
plt.grid(True, which='both')
plt.show()

