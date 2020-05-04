import numpy as np
import csv
import matplotlib.pyplot as plt

def ScatterPlot(x_array, y_array, title='Scatter Plot',
                x='Independent Variable', 
                y='Dependent Variable',
                save=False):
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.scatter(x_array, y_array, c='blue', marker='H')
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
    return x, y

def loadDAT(file, x_col=0, y_col=1):
    # Using with open built-in command
    results = open(file, 'r')
    x_list = list()
    y_list = list()
    for line in results:
        fields = line.split('\t')
        x_list.append( fields[0] )
        y_list.append( fields[1] )

    x = np.zeros( len(x_list)-1 )
    y = np.zeros( len(x_list)-1 )

    X = x_list[1:]
    Y = y_list[1:]

    for i in range( len(X) ):
        x[i] = float( X[i] )
        y[i] = float( Y[i] )
    
    return x, y

if __name__ == '__main__':
    name_txt = '../data/snow.txt'
    name_csv = '../data/snow.csv'
    name_dat = '../data/snow.dat'
    # x_load, y_load = loadTXT(name_txt)
    # x_load, y_load = loadCSV(name_csv)
    x_load, y_load = loadDAT(name_dat)
    ScatterPlot(x_load, y_load)
