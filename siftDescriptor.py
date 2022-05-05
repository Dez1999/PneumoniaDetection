import numpy as np 
import cv2 
from sklearn.svm import SVC
import matplotlib.cm as cm
import pandas as pd


#Author: Desmond Blake

#Method: This file is used to extract features descriptors from images in order to properly classify each
# type of image. Extract features with SIFT (Sacle Invariant Feature Transform)

#Method: Used to extract the descriptors from each image in the dataset
def getSiftFeatures(images_data):
    #Create Sift Detector (Scale Invariant Feature Transform)
    detector = cv2.SIFT_create()

    #Initialize list of descriptors for images
    descriptors_List = []

    #Find and compute the keypoints and descriptors using SIFT
    for img, label in images_data:
        keypoint, descriptor = detector.detectAndCompute(img, None)
        descriptors_List.append(descriptor)

    #After computing keypoints and descriptors, return the list of descriptors for each image
    return descriptors_List


#Method: Collects the individual descriptors in the list of descriptors and returns a new list containing all the descriptors
def computeAllDescriptors(descriptors_List):
    main_descriptors = []
    for indDescriptor in descriptors_List:
        if indDescriptor is not None:
            for descriptor in indDescriptor:
                main_descriptors.append(descriptor)

    return main_descriptors



