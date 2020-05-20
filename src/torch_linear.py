import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import dataload

class TorchLinearRegression(object):
    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def training_session(self, epochs=100, learn=0.001):
        model       = torch.nn.Linear(1,1)
        criterion   = torch.nn.MSELoss()
        optimizer   = torch.optim.SGD(model.parameters(), lr=learn)
        
        for epoch in range(epochs):
            outputs = model( torch.from_numpy(self.x_train) )
            losses  = criterion( outputs, torch.from_numpy(self.y_train) )
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            if (epoch+1) % 10 == 0:
                print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs,
                losses.item()))
        return model         

    def evaluation_session(self, model):
        validation = model( torch.from_numpy(self.x_train) ).detach().numpy()
        m_pred = (validation[1]-validation[0]) / (self.x_train[1]-self.x_train[0])
        b_pred = validation[0] - m_pred*self.x_train[0]
        return m_pred, b_pred, validation

    def testing_session(self, model, new_var):
        new_var = Variable(torch.Tensor( [new_var] ))
        prediction_result = model(new_var).item()
        return prediction_result

if __name__ == '__main__':
    # snowflakes = '../data/test/test.csv'
    snowflakes = '../data/snow/snow.csv'
    x_train, y_train = dataload.loadCSV(snowflakes)
    x_train = x_train.astype('float32')
    y_train = y_train.astype('float32')

    test_linear = TorchLinearRegression(x_train, y_train)
    myModel = test_linear.training_session(1000, 0.001)
    # check_results = test_linear.testing_session(myModel, 16)
    # print(check_results)

    # Plot the graph
    m_pred, b_pred, predictionSession = test_linear.evaluation_session(myModel)
    print("Predicted Slope: {}".format(m_pred))
    print("Predicted Intercept: {}".format(b_pred))
    # print(predictionSession)

    plt.plot(x_train, y_train, 'ro', label='Original data')
    plt.plot(x_train, predictionSession, label='Fitted line')
    plt.xlabel('Independent Variable')
    plt.ylabel('Dependent Variable')
    plt.title('Plotting with Machine Learning')
    plt.show()
