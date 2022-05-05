import os
import numpy as np 
import cv2
import pickle

#File: This file is used to serialize the image data into a pickle file
#Author: Desmond Blake

def loadAndLabelData(image_Dataset, classes):
    #Initialize all data for images
    allData = []

    #Iterate through each category and choose the selected training set
    for class_type in classes:
        #Initialize path for each class type
        path = os.path.join(image_Dataset, class_type)
        
        #Define labels for the different classes. Ex: 0 = dog, 1 = cat
        image_label = classes.index(class_type)

        #Iterate through each image in the class folder
        for image in os.listdir(path):
            #Identify an inidvidual image in the class folder
            imagepath = os.path.join(path, image)

            #Read the image 
            class_image = cv2.imread(imagepath, 0)
            try:
                #Resize all images to be the same size (100,100)
                class_image_ = cv2.resize(class_image, (100, 100))
                #image_resize = np.array(class_image).flatten()
                image_final = cv2.cvtColor(class_image_, cv2.COLOR_BGR2RGB)

                #Add image and image label to the array containing all the data
                allData.append([image_final, image_label])

            except Exception as e:
                pass
    return allData

def serializeObjects(outputFileName, data):
    #Serialize image objects
    print("Serializing Data of Images and Image Labels")
    fileName = outputFileName + '.pickle'
    pickle_out = open(fileName, 'wb')
    pickle.dump(data, pickle_out)
    pickle_out.close()


