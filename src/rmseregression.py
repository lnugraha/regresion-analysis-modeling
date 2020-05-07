import numpy as np
import pandas as pd
from random import seed, randrange
import math

def cross_validation_split(dataset, n_folds):
    cross_validation_dataset = []
    dataset_copy = list(dataset)
    # Round down please
    fold_size = int(len(dataset_copy)/n_folds)

    for _ in range(0, n_folds):
        fold = []
        while len(fold) < fold_size:
            index = randrange( len(dataset_copy) )
            fold.append( dataset_copy.pop(index) )
        cross_validation_dataset.append(fold)
    return cross_validation_dataset

def RMSE(actual_y, predicted_y):
    total_error = 0.0
    for i in range(0, len(actual_y)):
        error = predicted_y[i] - actual_y[i]
        total_error += error**2
    mean_error = total_error/float( len(actual_y) )
    return mean_error

def predict(row, coefficients):
    for i in range(0, len(row)-1):
        y_hat = coefficients[i+1]*row[i] + coefficients[0]
    return y_hat

def determine_sgd_coefficient(training_set, n_epoch, learning_rate=0.001):
    coefficients = [0.0 for _ in range(0, len(training_set[0]))]
    for _ in range(0, n_epoch):
        for row in training_set:
            y_hat = predict(row, coefficients)
            error = y_hat - row[-1]
            # coefficients[0] = coefficients[0] - learning_rate*error
            coefficients[0] -= learning_rate*error
            for i in range(0, len(row)-1):
                # coefficients[i+1] = coefficients[i+1] - learning_rate*error*row[i]
                coefficients[i+1] -= learning_rate*error*row[i]

    return coefficients

def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = []
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = []
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        rmse = RMSE(actual, predicted)
        scores.append(rmse)
    return scores

def linear_regression_sgd(train, test, learning_rate=0.001, n_epoch=10):
    predictions = []
    coefficients = determine_sgd_coefficient(train, n_epoch, learning_rate)
    for row in test:
        y_hat = predict(row, coefficients)
        predictions.append(y_hat)
    return predictions

if __name__ == '__main__':
    dataset = pd.read_csv('winequality_white.csv')

    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    dataset = sc.fit_transform(dataset)
    dataset = dataset.tolist()

    n_folds = 4
    learning_rate = 0.001
    n_epoch = 100
    scores = evaluate_algorithm(dataset, linear_regression_sgd, n_folds)

    print('Scores    : %s' % scores)
    print('Mean RMSE : %.4f' % (sum(scores)/float(len(scores))))
