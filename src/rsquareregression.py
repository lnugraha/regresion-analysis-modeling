import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import style
from statistics import mean

style.use('fivethirtyeight')

def create_dataset(number_of_points, variance, step=2, correlation=False):
    val = 1
    ys = []
    for _ in range(number_of_points):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation=='pos':
            val += step
        elif correlation and correlation=='neg':
            val -= step

    xs = [i for i in range( len(ys) )]
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)

def best_fit_slope_and_intercept(xs, ys):
    m = ( (mean(xs)-mean(ys)) - mean(xs*ys) ) / ( mean(xs)**2 - mean(xs**2) )
    b = mean(ys) - m*mean(xs)
    return m, b

def squared_error(ys_original, ys_line):
    return sum( (ys_original-ys_line)**2 ) 

def coefficient_of_determination(ys_original, ys_line):
    y_mean_line = [mean(ys_original) for y in ys_original]
    squared_error_regression_line = squared_error(ys_original, ys_line)
    squared_error_y_mean = squared_error(ys_original, y_mean_line)

    return ( 1.0 - (squared_error_regression_line/squared_error_y_mean) )

if __name__ == '__main__':
    xs, ys = create_dataset(40, 40, 2, correlation='pos')
    m, b = best_fit_slope_and_intercept(xs, ys)

    regression_line = [m*x + b for x in xs]
    r_squared = coefficient_of_determination(ys, regression_line)
    
    print("Slope and Intercept: {} and {}".format(m, b))
    print("R-Squared: %.4f " % r_squared)

    plt.scatter(xs, ys, color='red', label='Original Data')
    plt.plot(xs, regression_line, label = 'Regression Line')
    plt.legend(loc=4)
    plt.show()
