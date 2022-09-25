import random
import numpy as np
import matplotlib.pyplot as plot

from . import *
from Problem3 import *

import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
        
    np_data = np.genfromtxt('iris.data', delimiter=',')
    np_data = np.delete(np_data, 4, axis=1)
    
    sepal_proportions = np_data[:, 0] / np_data[:, 1]
    petal_proportions = np_data[:, 2] / np_data[:, 3]
    
    iris_proportions= np.array([sepal_proportions, petal_proportions]).T
    
    accuracies = []
    accuracies1 = []
    new_centers1 = []
    
    for k in range(1, 6):
        new_centers = k_means_pp(iris_proportions, k, 50)
        accuracies.append(compute_objective(iris_proportions, new_centers))
    
    for iter in range(1, 100):
        new_centers1 = k_means_pp(iris_proportions, 3, iter)
        accuracies1.append(compute_objective(iris_proportions, new_centers1))
    
    clusters = assign_data2clusters(iris_proportions, new_centers1)
    labels = np.argmax(clusters, axis=1)
    
    plot.plot(np.arange(1, 6), accuracies)
    plot.ylabel('Clustering Objective')
    plot.xlabel('Number of Clusters')
    plot.show()
    
    plot.plot(np.arange(1, 100), accuracies1)
    plot.ylabel('Objective')
    plot.xlabel('Iterations')
    plot.show()

    plot.scatter(x=iris_proportions[:, 0], y=iris_proportions[:, 1], c=labels)
    plot.xlabel('Sepal Proportion')
    plot.ylabel('Petal Proportion')
    plot.show()