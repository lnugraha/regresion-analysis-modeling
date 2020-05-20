import dataload
import matplotlib.pyplot as plt
import numpy as np
from random import seed, randrange
import os, sys, glob

class LINEAR_REGRESSION(object):
    pass
    # A master class that can plot the scatter plot and regression line
    # Same constructor

class LINEAR_RSQUARE(object):
    def __init__(self, xs, ys):
        self.xs = xs 
        self.ys = ys
        self.size = len(xs)
        if (self.size == 0):
            print("ERROR: Data size cannot be zero")
            sys.exit()
        elif (len(xs) != len(ys)):
            print("ERROR: Independent and dependent variable length is unequal")
            sys.exit()

    def slope_intercept(self):
        n = self.size
        S_x  = sum( self.xs ); S_y  = sum( self.ys )
        S_xx = sum( self.xs**2 ); S_xy = sum( self.xs*self.ys ) 
        # S_yy = sum( ys**2 )

        m = (n*S_xy - S_x*S_y) / (n*S_xx - S_x**2)
        b = S_y/n - m*(S_x/n)
        return m, b
    
    def squared_error(self, original, estimation):
        sum_sqr_err = 0.0
        for i in range( len(original) ):
            sqr_err = (estimation[i] - original[i])**2
            sum_sqr_err += sqr_err
        return sum_sqr_err

    def rsquared(self):
        ys_original  = list(self.ys)
        m_est, b_est = self.slope_intercept()

        ys_estimation = [m_est * x + b_est for x in self.xs]   
        ys_mean_line  = [np.mean(self.ys) for y in ys_original] 
        SS_tot = self.squared_error(ys_original, ys_mean_line)
        SS_res = self.squared_error(ys_original, ys_estimation)
        return (1.0 - (SS_res/SS_tot))
    
class LINEAR_RMSE(object):
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys

    def cross_validation(self, dataset, folds):
        cross_validation_dataset = list()
        dataset_copy = list(dataset)
        fold_size = int( len(dataset_copy)/folds )

        for _ in range (folds):
            fold = list()
            while len(fold) < fold_size:
                index = randrange( len(dataset_copy) )
                fold.append( dataset_copy.pop(index) )
            cross_validation_dataset.append(fold)
        return cross_validation_dataset

    def RMSE(self, y_original, y_estimation):
        sum_error = 0.0
        for i in range(0, len(y_original)):
            error = y_estimation[i] - y_original[i]
            sum_error += error**2
        mean_error = sum_error/float( len(y_original) )
        return mean_error

    def predict(self, row, coeff):
        for i in range ( len(row)-1 ):
            y_hat = coeff[i+1]*row[i] + coeff[0]
        return y_hat

    def SGD_coeff(self, train_set, epoch=5, lr=0.001):
        coeff = [0.0 for _ in range ( len(train_set[0]) )]
        for _ in range (epoch):
            for row in train_set:
                y_hat = self.predict(row, coeff)
                error = y_hat - row[-1]
                coeff[0] -= lr * error
                for i in range( len(row)-1 ):
                    coeff[i+1] -= lr*error*row[i]
        return coeff

    def evaluate_algorithm(self, dataset, algorithm, folds):
        pass

    def linear_regression_sgd(self, train, test, lr=0.001, epoch=5):
        pass

if __name__ == '__main__':
    snow_data = '../data/snow/snow.csv'
    x_snow, y_snow = dataload.loadCSV(snow_data)
    linear_reg = LINEAR_RSQUARE(x_snow, y_snow)
    m_snow, b_snow = linear_reg.slope_intercept()
    regression_line = [m_snow * x + b_snow for x in x_snow]
    rsqr = linear_reg.rsquared()

    print("Slope and Intercept: {} and {}".format(m_snow, b_snow) )
    print("R-square: {}".format(rsqr))
    
    plt.scatter(x_snow, y_snow, color='red')
    plt.plot(x_snow, regression_line, color='blue')
    plt.title('Plot of Snow vs Yield using Statistical Analysis')
    plt.xlabel('Snow')
    plt.ylabel('Yield')
    plt.show()
