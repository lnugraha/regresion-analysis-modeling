import numpy as np
import math, csv, sys, os

import matplotlib.pyplot as plt
# import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from abc import ABC, abstractmethod
from collections import Counter

def onlyNumber(dataIN):
    """
    Perfom data check; reject all complex or non-numbers (boolean or strings)
    Raise Type Error : if dataIN is not a number
    Raise Value Error: if dataIN is not a real number
    """
    if type(dataIN) not in [int, float, np.float64, np.float32, 
        np.int8, np.int16, np.int32, np.int64,
        np.uint8, np.uint16, np.uint32, np.uint64]:
        raise TypeError("Input data must be REAL numbers; data type is {}"
            .format(type(dataIN)))
    elif type(dataIN) in [complex, np.complex64, np.complex128]:
        raise ValueError("Input data cannot be COMPLEX numbers")
    else:
        pass

def ColoredScatterPlot(x_array, y_array, z_array,
                       title='Colored Scatter Plot',
                       x='Independent Variable', y='Dependent Variable',
                       save=False):
    pass
    # Coming Soon
    # TODO: Check the length of z_array then decide how many colors will be 
    # provided
    # z_array provide an extra dimensionality through color; hence the value 
    # MUST be discrete
    
    unique = Counter( z_array ).keys()
    if (unique != 2):
        print("Only binary classification is supported at the moment")
        sys.exit
    # Classify based on unique keys 

def ScatterPlot(x_array, y_array, title='Scatter Plot',
                x='Independent Variable', 
                y='Dependent Variable',
                save=False):
    plt.title(title); plt.xlabel(x); plt.ylabel(y)
    plt.scatter(x_array, y_array, c='blue', marker='H')
    if save == False:
        plt.show()
    elif save == True:
        plt.savefig('{}.png'.format(title))

def SurfacePlot(x_array, y_array, function_model,
                title='Surface Plot',
                x='Independent Variable 1', y='Independent Variable 2',
                z='Dependent Variable', 
                save=False):
    # TODO: Check Please
    x_array, y_array = np.meshgrid(x_array, y_array)
    z_array = function_model(x_array, y_array)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x_array, y_array, z_array, cmap=cm.coolwarm,
        linewidth=0, antialiased=False)
    if save == False:
        plt.show()
    elif save == True:
        plt.savefig('{}.png'.format(title))

def loadTXT(file, x_col=0, y_col=1):
    """
    Load a text file and returned as both independent and dependent arrays
    Inputs:
    file : .txt file
    x_col: independent variable column pos
    y_col: dependent variable column pos
    Outputs: x-array and y-array
    """
    # Load .txt data using np methods
    results = np.loadtxt(file, comments='#', delimiter='\t')
    # Independent Variables (Column 0)
    x = np.array( results[:, x_col], dtype=float ) 
    # Dependent Variables (Column 1)
    y = np.array( results[:, y_col], dtype=float ) 
    # Assertion
    for i in range( len(x) ):
        onlyNumber( x[i] ); onlyNumber( y[i] )
    
    return x, y

def loadCSV(file, x_col=0, y_col=1):
    # Using csv, delimiter is a comma sign (,)
    reader = csv.reader( open(file, 'r'), delimiter=',' )
    result = list(reader)

    rows = len(result) #including header
    cols = len(result[0])

    x = np.ndarray(rows, np.float64)
    y = np.ndarray(rows, np.float64)

    for i in range(rows):
        if (i != 0): # Skip the header part
            x[i] = result[i][x_col]
            y[i] = result[i][y_col]

    x = x[1:] # Decimate the very first unused element
    y = y[1:] # Decimate the very first unused element
    # Assertion
    for i in range ( len(x) ):
        onlyNumber( x[i] ); onlyNumber( y[i] )

    return x, y

def loadDAT(file, x_col=0, y_col=1):
    # Using with open built-in command
    results = open(file, 'r')
    x_list = list(); y_list = list()
    for line in results:
        fields = line.split('\t')
        x_list.append( fields[0] )
        y_list.append( fields[1] )

    x = np.zeros( len(x_list)-1 ); y = np.zeros( len(x_list)-1 )
    X = x_list[1:]; Y = y_list[1:]

    for i in range( len(X) ):
        x[i] = float( X[i] ); y[i] = float( Y[i] )
        onlyNumber(x[i]); onlyNumber(y[i])
    
    return x, y

class LoadDIM(ABC):
    """
    Load a data file and return THREE different arrays simultaneously
    Capable of handling multidimensional data (three dimensional)
    Only handle .csv and .txt data extension
    """
    def __init__(self, filein):
        self.filein = filein

    @abstractmethod
    def extractDIM(self, filein, col0=0, col1=1, col2=2):
        pass

class loadTXT_DIM(LoadDIM):
    def __init__(self, filein):
        self.filein = filein
    
    def extractDIM(self, col0=0, col1=1, col2=2):
        results = np.loadtxt( self.filein, comments='#', delimiter='\t')
        x = np.array( results[:,col0] )
        y = np.array( results[:,col1] )
        z = np.array( results[:,col2] )
        for i in range (len(x)):
            onlyNumber(x[i]); onlyNumber(y[i]); onlyNumber(z[i])
        return x, y, z

class loadCSV_DIM(LoadDIM):
    def __init__(self, filein):
        self.filein = filein
    
    def extractDIM(self, col0=0, col1=1, col2=2):
        # Using csv, delimiter is a comma sign (,)
        reader = csv.reader( open(self.filein, 'r'), delimiter=',' )
        result = list(reader)
        rows = len(result) #including header

        x = np.zeros(rows); y = np.zeros(rows);z = np.zeros(rows)

        for i in range(rows):
            if (i != 0): # Skip the header part
                x[i] = result[i][col0]
                y[i] = result[i][col1]
                z[i] = result[i][col2]

        x = x[1:]; y = y[1:]; z = z[1:] 
        for i in range (len(x)):
            onlyNumber(x[i]); onlyNumber(y[i]); onlyNumber(z[i])
        return x, y, z

def function_model(x_array, y_array):
    # Complete arbitrary function model here
    z_array = np.sin( np.sqrt(x_array**2 + y_array**2) )
    return z_array
    
if __name__ == '__main__':
    name_txt = '../data/snow/snow.txt'
    name_csv = '../data/snow/snow.csv'
    name_dat = '../data/snow/snow.dat'

    duration_csv = '../data/duration/duration.csv'
    wblake = '../data/triplet/wblake.txt'
    triplets = '../data/triplet/svmsample.csv'

    # x_load, y_load = loadTXT(name_txt)
    # x_load, y_load = loadCSV(duration_csv)
    # x_load, y_load = loadDAT(name_dat)
    
    # multidim_data = loadCSV_DIM(triplets)
    # x_load, y_load, z_load = multidim_data.extractDIM()
    
    multidim_data = loadTXT_DIM(wblake)
    x_load, y_load, z_load = multidim_data.extractDIM()
    
    # perceptron = '../data/triplet/perceptron.csv'
    # multidim_data = loadCSV_DIM(perceptron)
    # x_load, y_load, z_load = multidim_data.extractDIM()

    # ScatterPlot(y_load, z_load)
    # ScatterPlot(x_load, y_load)

    x_array = np.arange(-5, 5, 0.1)
    y_array = np.arange(-5, 5, 0.1)
    z_array = np.zeros( len(x_array) )
    SurfacePlot(x_array, y_array, function_model)