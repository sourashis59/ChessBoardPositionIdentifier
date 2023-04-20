# ======================================OBJECTIVE====================================
#
# predict output for data using the models in "trainedModel.p"
# 
# ===================================================================================



# from PIL import Image
# from skimage.io import imread
# from skimage.transform import resize
# import os
# import numpy as np
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import train_test_split, GridSearchCV
# # from sklearn.svm import SVC
# import pickle



# from PIL import Image
import tensorflow as tf
import os 
import cv2
# import imghdr
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.metrics import Precision, Recall , BinaryAccuracy





use_tf_keras_models_evaluate = False





resizedImageDimension = (64, 64)





filePath = 'data/testData/pieceTypeClassifier'
filePathForEvaluate = 'data/testData/pieceTypeClassifierForEvaluate'


# classifier = pickle.load(open('models/pieceTypeClassifier/trainedModel.p', 'rb'))
classifier = load_model( os.path.join("models/pieceTypeClassifier/trainedModelCNN.h5") )



classes = ['bishop', 'king', 'knight', 'pawn', 'queen', 'rook']

data = []
# images = []
imageFileNames = []


print("\n\n\nCollecting data......................")











if(not use_tf_keras_models_evaluate):

    
    for file in os.listdir(filePath):
        imagePath = os.path.join(filePath, file)
        imageFileNames.append(file)
        # images.append(Image.open(imagePath))

        # image = Image.open(imagePath)
        # images.append(image)
        # image = image.resize((20,20))
        # data.append( list(image.getdata()) )


        # image = imread(imagePath)
        
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = tf.image.resize(image, resizedImageDimension).numpy().astype(int)
        
        # plt.imshow(image)    
        # plt.show()

        data.append(image)

    print("Data Collected!\n")




    print(f"images len = {len(data)}")
    # images[40].show()

    wrongGuessCount = 0
    allVerdictCorrect = True
    wrongGuessFiles = []
    wrongGuesses = []

    totalData = len(data)

    print("\n\ntesting data.......................................\n")
    for i in range(len(data)):

        if(i % 500 == 0):
            print(f"Tested data : {i}/{totalData},   wrong guess : {wrongGuessCount}")

        # images[i].show()
        
        # for the CNN model, we need batch of images. 
        transformedData = np.expand_dims(data[i]/255, 0)
        # print(transformedData)
        
        # plt.imshow(transformedData[0])
        # plt.show()

        # # pred = classifier.predict([data[i]])

        #* it will return batch(of size 1) of predicted probabilities of classes(2 classes here)
        #* for example : pred = [[9.999664e-01 3.858412e-06]] 
        # *verbose = 0 ====> dont print anything while doing prediction
        pred = classifier.predict(transformedData, verbose=0) 
        # print(pred)

        verdict = "NULL"

        predClassInd = np.argmax(pred[0])    


        if( imageFileNames[i].startswith(classes[predClassInd]) ) :
            verdict = "correct" 
        else :
            verdict = "wrong"    
            allVerdictCorrect = False
            wrongGuessCount += 1

            # plt.imshow(transformedData[0])
            # plt.show()

            wrongGuessFiles.append(imageFileNames[i])
            wrongGuesses.append(classes[predClassInd])

        # print(f"i = {i}, { classes[predClassInd] },  verdict = {verdict}")    
        







    print(f"\nWrongly guessed files : ")
    for i in range(len(wrongGuessFiles)):
        print(f"file : {wrongGuessFiles[i]},    wrongGuess = {wrongGuesses[i]}")
            
    print("----------------------------------")




    if(allVerdictCorrect) :
        print("\nAll guesses are correct\n")
    else :
        print("\nAll guesses are NOT correct!!!!!!!!!!!!!!!!!!!!!\n")


    print(f"\nwrong guesses : {wrongGuessCount}")
    print(f"Accuracy : {1 - (wrongGuessCount / totalData)}")







# else :
    testData = tf.keras.utils.image_dataset_from_directory(filePathForEvaluate, image_size=resizedImageDimension)
    testData = testData.map(lambda x,y : (x/255, y)) 

    
    testingResult = classifier.evaluate(testData)
    print("\n\nTesting result (using evaluate() ): ")
    print(testingResult)
    print("\n\n\n\n\n")
