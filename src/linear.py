import dataload
import matplotlib.pyplot as plt
import numpy as np
from statistics import mean

class LINEAR_RSQUARE(object):
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys

    def slope_intercept(self, xs, ys):
        m = ( (mean(xs)-mean(ys)) - mean(xs*ys) ) / (mean(xs)**2 - mean(xs**2))
        b = mean(ys) - m*mean(xs)
        return m, b
    
    def squared_error(self, original, estimation):
        sum_sqr_err = 0.0
        for i in range(len(original)):
            sqr_err = (estimation[i] - original[i])**2
            sum_sqr_err += sqr_err
        return sum_sqr_err

    def rsquare(self, ys):
        
        ys_original = list(self.ys)
        m_est, b_est = self.slope_intercept(self.xs, self.ys)

        ys_estimation = [m_est * x + b_est for x in self.xs]   # np.array
        y_mean_line = [mean(ys_original) for _ in ys_original] # np.array
        
        squared_error_reg_line = self.squared_error(ys_original, ys_estimation)
        squared_error_y_mean = self.squared_error(ys_original, y_mean_line)
        
        # r-square value does not look right
        return (1.0 - (squared_error_reg_line/squared_error_y_mean))
    
class LINEAR_RMSE(object):
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys

    def RMSE(self, y_original, y_estimation):
        sum_error = 0.0
        for i in range(0, len(y_original)):
            error = y_estimation[i] - y_original[i]
            sum_error += error**2
        mean_error = sum_error/float( len(y_original) )
        return mean_error

    def predict(self):
        pass

    def SGD_coeff(self):
        pass

if __name__ == '__main__':
    snow_data = '../data/snow/snow.csv'
    x_snow, y_snow = dataload.loadCSV(snow_data)
    linear_reg = LINEAR_RSQUARE(x_snow, y_snow)
    m_snow, b_snow = linear_reg.slope_intercept(x_snow, y_snow)
    regression_line = [m_snow * x + b_snow for x in x_snow]
    rsqr = linear_reg.rsquare(y_snow)

    print("Slope and Intercept: {} and {}".format(m_snow, b_snow) )
    print("{}".format(rsqr))
    plt.scatter(x_snow, y_snow, color='red')
    plt.plot(x_snow, regression_line, color='blue')
    plt.show()



