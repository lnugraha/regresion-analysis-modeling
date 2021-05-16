import numpy as np
import matplotlib.pyplot as plt
import dataload

def get_weights(x, y, verbose=0):
    shape = x.shape
    x = np.insert(x, 0, 1, axis=1)
    w = np.ones((shape[1]+1,))
    weights = []

    learning_rate = 10; iteration = 0; loss = None

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
    
    """
    df = np.loadtxt("../data/triplet/perceptron.csv", delimiter = ',')
    x  = df[:,0:-1] # x is [ [], [], [], [] ]
    y  = df[:,-1]   # y is [ 1, 1, -1, -1 ]

    """
    multidim_data = dataload.loadCSV_DIM('../data/triplet/svm_test.csv')
    x_load, y_load, z_load = multidim_data.extractDIM()
    y = z_load
    x = np.concatenate( (x_load, y_load) )
    x = np.reshape(x, (len(x_load),2), order='F')

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
    


    
    # plt.scatter(x[y>0][:,1], x[y>0][:,2], marker = 'x')
    # plt.scatter(x[y<0][:,1], x[y<0][:,2], marker = 'o')
    

    """
    for i in range (len(x_load)):
        if (z_load[i] == 1.0):
            plt.scatter(x_load[i], y_load[i], c='red')
        elif (z_load[i] == -1.0):
            plt.scatter(x_load[i], y_load[i], c='blue')
    plt.plot(x1, x2) # Separator
    plt.title('Perceptron Seperator')
    plt.xlabel('Feature 1 ($x_1$)')
    plt.ylabel('Feature 2 ($x_2$)')
    plt.show()
    """

    counter = 0
    for ix, w in enumerate(all_weights):
        if ix % 10 == 0:
            print('Weights:', w)
            x1 = np.linspace(np.amin(x[:,1]),np.amax(x[:,2]),2)
            x2 = np.zeros((2,))
            for ix, i in enumerate(x1):
                x2[ix] = (-w[0] - w[1]*i) / w[2]
            print('$0 = ' + str(-w[0]) + ' - ' + str(w[1]) + 'x_1'+ ' - ' + 
                str(w[2]) + 'x_2$')

            # plt.scatter(x[y>0][:,1], x[y>0][:,2], marker = 'x')
            # plt.scatter(x[y<0][:,1], x[y<0][:,2], marker = 'o')
            
            plt.clf()
            for i in range (len(x_load)):
                if (z_load[i] == 1.0):
                    plt.scatter(x_load[i], y_load[i], c='red')
                elif (z_load[i] == -1.0):
                    plt.scatter(x_load[i], y_load[i], c='blue')
            
            plt.plot(x1,x2)
            plt.title('Perceptron Separator after ' + str(counter) + ' Iterations')
            plt.xlabel('Feature 1 ($x_1$)')
            plt.ylabel('Feature 2 ($x_2$)')
            # plt.show()
            plt.savefig('Iteration Number ' + str(counter) + '.png')
        counter += 1
