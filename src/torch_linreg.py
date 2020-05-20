import torch
from torch.autograd import Variable

x_data = Variable(torch.Tensor([[1.0],[2.0],[3.0], [4.0], [5.0]]))
y_data = Variable(torch.Tensor([[2.0],[4.0],[6.0], [8.0], [10.0]]))

class LinearRegression(torch.nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(1,1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

test_model = LinearRegression()

criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD( test_model.parameters(), lr = 0.001 )

for epoch in range(1000):
    pred_y = test_model(x_data)
    loss = criterion(pred_y, y_data)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch % 100 == 0):
        print('Epoch {}, Loss {}'.format(epoch, loss))

new_var = Variable(torch.Tensor([[12.0]]))
pred_y = test_model(new_var)
print("Prediction after training", 12, test_model(new_var).item() )
