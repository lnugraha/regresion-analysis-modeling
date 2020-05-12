import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

import dataload

# Hyper-parameters
input_size = 1
output_size = 1
num_epochs = 100
learning_rate = 0.001

# Toy dataset
snowflakes = '../data/snow/snow.csv'
x_train, y_train = dataload.loadCSV(snowflakes)
x_train = x_train.astype('float32')
y_train = y_train.astype('float32')

# Linear regression model
model = nn.Linear(input_size, output_size)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  

# Train the model
for epoch in range(num_epochs):
    # Convert numpy arrays to torch tensors
    inputs = torch.from_numpy(x_train)
    targets = torch.from_numpy(y_train)

    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 5 == 0:
        print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

# Plot the graph
predicted = model(torch.from_numpy(x_train)).detach().numpy()
plt.plot(x_train, y_train, 'ro', label='Original data')
plt.plot(x_train, predicted, label='Fitted line')
# plt.legend()
plt.xlabel('Snow')
plt.ylabel('Yield')
plt.title('Plot of Snow vs Yield using Machine Learning')
plt.show()
