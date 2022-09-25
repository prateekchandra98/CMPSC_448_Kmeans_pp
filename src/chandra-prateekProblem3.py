import numpy as np
#import scipt as sp
import matplotlib.pyplot as plt
from numpy import random

def k_init(X, k):
    """ k-means++: initialization algorithm

    Parameters
    ----------
    X: array, shape(n ,d)
        Input array of n samples and d features

    k: int
        The number of clusters

    Returns
    -------
    init_centers: array (k, d)
        The initialize centers for kmeans++
    """
    init_centers = []
    center = [random.randint(0, X.shape[0]-1)]
    init_centers.append(X[center[0]])
    
    #looping through k clusters
    for i in range(1, k):
        #this array stores the distance of each point to the first cluster center
        distances = np.array([]) 
        #this array stores the probability
        probability = []
        #summation of all the distances to calculate the probability
        summation = 0 
        #looping through each data point to calculate the distance from the center
        for j in range (0, X.shape[0]):
            #large value so that all the distances computed are smaller than this value
            minDistance = 999999999999;
            #looping through all the centers and finding the closest center to the test point
            for centers in range(0, len(init_centers)):
                distCalculation = np.linalg.norm(X[j]-init_centers[centers])
                #if the distance is less than the previous calculated distance, use the current distance
                if distCalculation < minDistance:
                    minDistance = distCalculation
            
            distances = np.append(distances, minDistance)
            summation = summation + minDistance
        
        probability = [x / summation for x in distances]
        
        index = np.random.choice(range(0,len(distances)), p=probability)
        init_centers.append(X[index])
    init_centers = np.asarray(init_centers)
    return init_centers


def k_means_pp(X, k, max_iter):
    """ k-means++ clustering algorithm

    step 1: call k_init() to initialize the centers
    step 2: iteratively refine the assignments

    Parameters
    ----------
    X: array, shape(n ,d)
        Input array of n samples and d features

    k: int
        The number of clusters

    max_iter: int
        Maximum number of iteration

    Returns
    -------
    final_centers: array, shape (k, d)
        The final cluster centers
    """
    #step 1: call k_init() to initialize the centers

    old_centers = k_init(X, k)
    final_centers = []
    
    #step 2: iteratively refine the assignments
    for iter in range(0, max_iter):
        data_map = assign_data2clusters(X,old_centers)
        final_centers = np.zeros(old_centers.shape)
        
        for center in range(0, len(old_centers)):
            num_pts = 0
            center_val_sum = np.zeros(X.shape[1])
            
            for point in range(0, X.shape[0]):
                if data_map[point,center] == 1:
                    center_val_sum += X[point]
                    num_pts += 1
            if num_pts > 0:
                final_centers[center]= center_val_sum / num_pts
            else:
                final_centers[center] = old_centers[center]
        old_centers = final_centers
    return final_centers

def assign_data2clusters(X, C):
    """ Assignments of data to the clusters
    Parameters
    ----------
    X: array, shape(n ,d)
        Input array of n samples and d features

    C: array, shape(k ,d)
        The final cluster centers

    Returns
    -------
    data_map: array, shape(n, k)
        The binary matrix A which shows the assignments of data points (X) to
        the input centers (C).
    """
    #declaration of array
    data_map = np.zeros((X.shape[0], len(C)), dtype=int)
    #[[0 for x in range(h)] for y in range(w)] 
    
    for point in range (0, (X.shape[0])):
        minDistance = 99999999999
        index = -1
        for center in range (0, (len(C))):
            distCalculation = np.linalg.norm(X[point]-C[center])
            #if the distance is less than the previous calculated distance, use the current distance
            if distCalculation < minDistance:
                minDistance = distCalculation
                index = center
        data_map[point,index] = 1
    return data_map

def compute_objective(X, C):
    """ Compute the clustering objective for X and C
    Parameters
    ----------
    X: array, shape(n ,d)
        Input array of n samples and d features

    C: array, shape(k ,d)
        The final cluster centers

    Returns
    -------
    accuracy: float
        The objective for the given assigments
    """
    accuracy = 0
    
    data_map = assign_data2clusters(X, C)
    #print(data_map)
    for point in range (0, X.shape[0]):
        for center in range(0, len(C)):
            if data_map[point][center] == 1:
                dist = np.linalg.norm(X[point] - C[center])
                accuracy += (dist*dist)
    return accuracy