from ast import Not
import numpy as np 
import cv2
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import pickle

#Author: Desmond Blake

#About This file is used to compute the bag of words dictionary used for image classification using an unisupervised methood called kMeans 
# to create kMean clusters. In addition, it is also used to compute the image features for each image class.


#Method: computeBagOfWords is used to compute visual words for classifying each individual image
def computeBagOfWords(total_clusters, descriptorsList):

    #Perform k-means clustering to partition all observations into k clusters
    k_means = KMeans(n_clusters = total_clusters)

    #Compute k-means
    k_means.fit(descriptorsList)

    #Initialize bag of visual words dictionary
    dictionary = []

    #Declare dictionary array as the coordinates of cluster centers
    dictionary = k_means.cluster_centers_

    #Save Bag of Words Dictionary
    print("Serializing Bag of Words Dictionary")
    fileName = "bow_dictionary" + '.pickle'
    pickle_out = open(fileName, 'wb')
    pickle.dump(dictionary, pickle_out)
    pickle_out.close()

    #Return computed Bag of Words Dictionary
    return dictionary


#Method: used to find the features using Bag of Words method and Kmeans clusters
def getBOWFeatures(bag_of_words, total_clusters, descriptors):

    max = len(descriptors)

    #Initialize features list
    main_features = []

    for n in range(max):
        #Set features array as an array for the total number of clusters
        print(total_clusters)
        features = np.array([0] * 400)

        if descriptors[n] is not None:
            #Compute distance between each pair of the two collection inputs
            print("Descriptors[n]")
            print(descriptors[n])
            print("Bag of Words")
            print(bag_of_words)
            dist = cdist(descriptors[n], bag_of_words)

            #Search for smallest possible values from descriptor and bag of words
            bin = np.argmin(dist,axis =1)

            #Add all features to the individual feature list
            for i in bin:
                features[i] += 1
        #Add features to the main feature list
        print("Next feature")
        main_features.append(features)
    
    #Return the computed feature list to use to train the model
    return main_features
    




    
