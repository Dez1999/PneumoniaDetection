import numpy as np 
import cv2
import matplotlib.pyplot as plt 
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
import matplotlib.cm as cm
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
from siftDescriptor import getSiftFeatures, computeAllDescriptors
from bagOfWordsComputation import computeBagOfWords, getBOWFeatures
#Model: This is the main file to run the image classification model
#Author: Desmond Blake

#Note: The below code, when uncommented, will take in different types of classes to classify, load and label the data
#and then serialize the image Objects into a pickle file for easier functionality and reduced running time

#Import custom files
from SerializeData import loadAndLabelData, serializeObjects

#Define path of dataset 
ImageDataSet = 'ImageDataset\\class_dataset'

#Define types of classes in the dataset
classes = ['NORMAL', 'PNEUMONIA']

#Save and Load data into an array
saved_data = loadAndLabelData(ImageDataSet, classes)
print("Saved Data")

#Serialize Image Objects
serializeObjects('data_images', saved_data)

# Load Pickled Image Data
pick_in = open('data_images.pickle', 'rb')
data = pickle.load(pick_in)
pick_in.close()

#USE SIFT DESCRIPTOR ALGORITHM
#Extract the sift descriptors from the image data
sift_descriptors = getSiftFeatures(saved_data)
print("Extracted Sift Descriptors")
print(sift_descriptors)

# #Collect each individual descriptor
main_descriptors = computeAllDescriptors(sift_descriptors)
print("Computed All Descriptors")
#print(main_descriptors)


#Collect and create a Bag of Visual Words to aid in classifying images
#Identify number of clusters to use based off of trial and error in order to figure out the best value
#to use for the number of clusters

cluster_size = 60
visual_Bag_of_Words = computeBagOfWords(cluster_size,main_descriptors)
print("Computed Bag of Words")

# print(visual_Bag_of_Words)
#Compute features with Bag of Words
features = []
features = getBOWFeatures(visual_Bag_of_Words, cluster_size, sift_descriptors)
print("Computed BOW features")


#Iterate through data and obtain labels
labels = []

for feature, label in saved_data:
    #features.append(feature)
    labels.append(label)


# Create training and testing data for image classification. Training and Testing datasets were created below
# and then used to train the model using Support Vector Machines.
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size = 0.20)
# print("Split Dataset into training and testing sets")

# After training the model, the model was saved in an 'sav' file that can easily be used for testing purposes
#Model
model = SVC(C=1, kernel='poly', gamma='auto')
model.fit(x_train, y_train)
print("Trained the model")

#Save Model
pick = open('ImageClassificationModel.sav', 'wb')
pickle.dump(model, pick)
pick.close()

# pick_inModel = open('imageClassificationModel.sav', 'rb')
# model = pickle.load(pick_inModel)
# pick_inModel.close()

#Predict the testing dataset of images
prediction = model.predict(x_test)

#Calculate the accuracy of the model
accuracy = model.score(x_test, y_test)

print("Accuracy: ", accuracy)
print("Accuracy Score: ", accuracy_score(y_test, prediction)*100)
print(classification_report(y_test, prediction))


# Plot non-normalized and normalized confusion matrix
options_titles = [
    ("Confusion matrix, Method: no normalization", None),
    ("Normalized confusion matrix, Method: normalization", 'true'),
]
for title, normalize in options_titles:
    disp = ConfusionMatrixDisplay.from_estimator(
        model,
        x_test,
        y_test,
        display_labels=classes,
        cmap=plt.cm.Greens,
        normalize=normalize,
    )
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.show()







