import matplotlib.pyplot as plt
from sklearn import linear_model
import dataload 

class Scikit_LinearRegression(object):
    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
    
    def fit(self):
        model = linear_model.LinearRegression()
        model.fit(x_train, y_train)
        return model

    def predict(self, model, x_train):
        y_predict = model.predict(x_train)
        return y_predict

if __name__ == '__main__':
    # snowflakes = '../data/test/test.csv'
    snowflakes = '../data/snow/snow.csv'
    x_train, y_train = dataload.loadCSV(snowflakes)
    x_train = x_train.astype('float32')
    y_train = y_train.astype('float32')

    snow_model = linear_model.LinearRegression()
    snow_model.fit(x_train, y_train)

    plt.scatter(x_train, y_train, c='r')
    plt.plot(x_train, snow_model.predict(x_train), label='Fitted Line')
    
    plt.xlabel('Independent Variable')
    plt.ylabel('Dependent Variable')
    plt.title('Plotting with Machine Learning')
    plt.show()
    
