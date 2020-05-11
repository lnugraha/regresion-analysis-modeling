import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Hyper-parameters
input_size = 1
output_size = 1
num_epochs = 100
learning_rate = 0.001

# Toy dataset
x_train = np.array([[23.1], [32.8], [31.8], [32.0], [30.4], [24.0], [39.5], 
                    [24.2], [52.5], [37.9], [30.5], [25.1], [12.4], [35.1], 
                    [31.5], [21.1], [27.6]], dtype=np.float32)

y_train = np.array([[10.5], [16.7], [18.2], [17.0], [16.3], [10.5], [23.1], 
                    [12.4], [24.9], [22.8], [14.1], [12.9], [8.8], [17.4],
                    [14.9], [10.5], [16.1]], dtype=np.float32)

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

# Save the model checkpoint
# torch.save(model.state_dict(), 'model.ckpt')
