#Gui model for computer vision predictions

import cv2
import streamlit as st
import pandas as pd
from PIL import Image
import pickle
import numpy as np 
from io import BytesIO
from bagOfWordsComputation import getBOWFeatures


def initializeGUI():
    html_temp = """
    <body style = "background-color:blue;">
    <div style ="background-color: teal ; padding: 10px">
    <h2 style="color:white; text-align:center;"> Pneumonia Detection Application - Image Classification</h2>
    </div>
    <div style ="background-color: white ; padding: 10px">
    <h5 style="color:black; text-align:center;"> This image classification model can be used to predict whether an X-ray image of a chest is Normal or contains signs of Pneumonia</h5>
    </div>
    <div style ="background-color: white ; padding: 10px">
    <h5 style="color:black; text-align:center;"> Accuracy: 89.81 %</h5>
    </div>
    </body>
    """

    st.markdown(html_temp, unsafe_allow_html=True)


    imageList = ["ImageDataset\\class_dataset\\NORMAL\\IM-0001-0001.jpeg","ImageDataset\\class_dataset\\PNEUMONIA\\person3_bacteria_10.jpeg"]
    captionList = ["Normal Chest X-ray","Pneumonia Chest X-ray"]

    #Reformat Example images to be displayed
    displayImagesList = []
    for img in imageList:

        # Convert image into numpy array
        image = cv2.imread(img, 0)

        # print(type(image))
        # print(image.shape)

        # Resize Image for model
        # print("2 - Resize Image")
        if(type(image) == type(None)):
            pass
        else:
            Image_resized = cv2.resize(image, (200, 200))
        
        displayImagesList.append(Image_resized)

    #Display Example Images
    col1, col2 = st.columns(2)
    col1.image(displayImagesList[0], caption=captionList[0], width=300)
    col2.image(displayImagesList[1], caption=captionList[1], width=300)

    uploaded_file = st.file_uploader("Please choose an X-ray file of a chest", type=['jpg', 'png', 'jpeg'])

    if uploaded_file is not None:
        #Read image file
        selected_image = Image.open(uploaded_file)
        st.text("Uploaded X-ray File")

        #Reformat and Display the Uploaded Image
        bytes_data_image = uploaded_file.getvalue()
        Image_array = np.array(Image.open(BytesIO(bytes_data_image)))
        Image_displayed = cv2.resize(Image_array, (300, 300))
        st.image(Image_displayed)

    
    if st.button("Predict"):
        #Preprocess Image
        final_Image  = processImage(uploaded_file)
        
        #Predict Image
        predictedResult = predictImage(final_Image)
        st.text("Prediction:" + predictedResult)


def processImage(image):
    # Obtain image bytes data
    bytes_data = image.getvalue()

    # Convert image into numpy array
    Image_array = np.array(Image.open(BytesIO(bytes_data)))
    # print("1 - Image Array ")
    # print(Image_array)
    #st.write(Image_array)

    # print(type(Image_array))
    # print(Image_array.shape)

    # Resize Image for model
    # print("2 - Resize Image")
    Image_resized = cv2.resize(Image_array, (100, 100))
    # print(Image_resized)

    # Convert image from BGR to RGB
    # print("3 - Convert to RGB")
    Image_RGB = cv2.cvtColor(Image_resized, cv2.COLOR_BGR2RGB)

    # st.image(Image_RGB)

    #Compute image Features
    image_features = getImageFeatures(Image_RGB)

    return image_features


def getImageFeatures(imageData):
    #Load BOW dictionary
    pick_inDictionary = open('bow_dictionary.pickle', 'rb')
    BOW_dictionary = pickle.load(pick_inDictionary)
    pick_inDictionary.close()

    #Get Sift Descriptors
    data = []
    detector = cv2.SIFT_create()
    keypoint, image_SIFT_descriptors = detector.detectAndCompute(imageData, None)
    data.append(image_SIFT_descriptors)

    #Obtain relevant image Features for Model
    clusterSize = 60
    imageFeatures = getBOWFeatures(BOW_dictionary, clusterSize, data)
    # print("Image Features")
    # print(imageFeatures)
    #Return Final Image Data and Features
    return imageFeatures



def predictImage(image):
    #Load pre-trained model
    pick_inModel = open('imageClassificationModel.sav', 'rb')
    model = pickle.load(pick_inModel)
    pick_inModel.close()

    #Predict image
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions, axis=-1)

    #Print Results
    # print(predictions)
    # print(predicted_class)

    if predictions[0] == 0:
        return "Normal"
    if predictions[0] == 1:
        return "Pneumonia"

    
initializeGUI()