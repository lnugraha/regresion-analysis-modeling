import dataload
import torch
from torch.autograd import Variable

class LinearRegression(torch.nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(1,1)
    
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

if __name__ == '__main__':
    
    datadir = '../data/snow/snow.csv'
    x_load, y_load = dataload.loadCSV(datadir)

    x_data = Variable( torch.Tensor(x_load.astype('float32')) )
    y_data = Variable( torch.Tensor(y_load.astype('float32')) )

    LinearModel = LinearRegression()
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(LinearModel.parameters(), lr=0.001)
    
    for epoch in range(1000):
        pred_y = LinearModel(x_data)
        loss = criterion(pred_y, y_data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch % 100 == 0):
            print('Epoch: {} -- Loss: {}'.format(epoch, loss))

    # pred_y = LinearModel(12)
