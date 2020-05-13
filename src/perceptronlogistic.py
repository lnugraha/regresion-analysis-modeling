import numpy as np
import matplotlib.pyplot as plt
import dataload

def get_weights(x, y, verbose=0):
    shape = x.shape
    x = np.insert(x, 0, 1, axis=1)
    w = np.ones((shape[1]+1,))
    weights = []

    learning_rate = 10
    iteration = 0
    loss = None
    while iteration <= 1000 and loss != 0:
        for ix, i in enumerate(x):
            pred = np.dot(i,w)
            if pred > 0: pred = 1
            elif pred < 0: pred = -1
            if pred != y[ix]:
                w = w - learning_rate * pred * i
            weights.append(w)    
            if verbose == 1:
                print('X_i = ', i, '    y = ', y[ix])
                print('Pred: ', pred )
                print('Weights', w)
                print('------------------------------------------')


        loss = np.dot(x, w)
        loss[loss<0] = -1
        loss[loss>0] = 1
        loss = np.sum(loss - y )

        if verbose == 1:
            print('------------------------------------------')
            print(np.sum(loss - y ))
            print('------------------------------------------')
        if iteration%10 == 0: learning_rate = learning_rate / 2
        iteration += 1    
    print('Weights: ', w)
    print('Loss: ', loss)
    return w, weights

if __name__ == '__main__':
    # df = np.loadtxt("perceptron.csv", delimiter = ',')
    x, y, z = dataload.loadCSV('../data/triplet/svm_test.csv')
    # x  = df[:,0:-1]
    # y  = df[:,-1]

    print('Dataset')
    print(df, '\n')

    w, all_weights = get_weights(x, y)
    x = np.insert(x, 0, 1, axis=1)

    pred = np.dot(x, w)
    pred[pred > 0] =  1
    pred[pred < 0] = -1
    print('Predictions', pred)

    x1 = np.linspace(np.amin(x[:,1]),np.amax(x[:,2]),2)
    x2 = np.zeros((2,))
    for ix, i in enumerate(x1):
        x2[ix] = (-w[0] - w[1]*i) / w[2]
    """
    plt.scatter(x[y>0][:,1], x[y>0][:,2], marker = 'x')
    plt.scatter(x[y<0][:,1], x[y<0][:,2], marker = 'o')
    plt.plot(x1,x2)
    plt.title('Perceptron Seperator', fontsize=20)
    plt.xlabel('Feature 1 ($x_1$)', fontsize=16)
    plt.ylabel('Feature 2 ($x_2$)', fontsize=16)
    plt.show()
    """
    for ix, w in enumerate(all_weights):
        if ix % 10 == 0:
            print('Weights:', w)
            x1 = np.linspace(np.amin(x[:,1]),np.amax(x[:,2]),2)
            x2 = np.zeros((2,))
            for ix, i in enumerate(x1):
                x2[ix] = (-w[0] - w[1]*i) / w[2]
            print('$0 = ' + str(-w[0]) + ' - ' + str(w[1]) + 'x_1'+ ' - ' + 
                str(w[2]) + 'x_2$')

            plt.scatter(x[y>0][:,1], x[y>0][:,2], marker = 'x')
            plt.scatter(x[y<0][:,1], x[y<0][:,2], marker = 'o')
            plt.plot(x1,x2)
            plt.title('Perceptron Seperator', fontsize=20)
            plt.xlabel('Feature 1 ($x_1$)', fontsize=16)
            plt.ylabel('Feature 2 ($x_2$)', fontsize=16)
            plt.show()
