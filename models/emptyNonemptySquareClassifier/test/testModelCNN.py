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



from PIL import Image
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





resizedImageDimension = (64, 64)
collabSourcePath = ""





# classifier = pickle.load(open('models/emptyNonemptySquareClassifier/trainedModel.p', 'rb'))
classifier = load_model( os.path.join(collabSourcePath, "models/emptyNonemptySquareClassifier/trainedModelCNN.h5") )

classes = ['emptySquare', 'nonemptySquare']

data = []
images = []
imageFileNames = []

filePath = 'models/emptyNonemptySquareClassifier/data/testData'

for file in os.listdir(filePath):
    imagePath = os.path.join(filePath, file)
    imageFileNames.append(file)
    images.append(Image.open(imagePath))

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



# data = np.asarray(data)

# nsamples, nx, ny = data.shape
# data = data.reshape((nsamples,nx*ny))



print(f"images len = {len(images)}")
# images[40].show()

wrongGuesses = 0
allVerdictCorrect = True
wrongGuessFiles = []

for i in range(len(data)):
    # images[i].show()
    
    # for the CNN model, we need batch of images. 
    transformedData = np.expand_dims(data[i]/255, 0)
    # print(transformedData)
    
    # plt.imshow(transformedData[0])
    # plt.show()

    # # pred = classifier.predict([data[i]])

    #* it will return batch(of size 1) of predicted probabilities of classes(2 classes here)
    #* for example : pred = [[9.999664e-01 3.858412e-06]] 
    pred = classifier.predict(transformedData) 
    # print(pred)

    verdict = "NULL"

    predSquareType = np.argmax(pred[0])
    # predSquareType = "NULL"
    # if(pred[0][0] < 0.5) :
    #     predSquareType = 0
    # else:
    #     predSquareType = 1         


    if(   imageFileNames[i].startswith(classes[predSquareType]) ) :
        verdict = "correct" 
    else :
        verdict = "wrong"    
        allVerdictCorrect = False
        wrongGuesses += 1

        # plt.imshow(transformedData[0])
        # plt.show()

        wrongGuessFiles.append(imageFileNames[i])

    print(f"i = {i}, { classes[predSquareType] },  verdict = {verdict}")    
    







print(f"\nWrongly guessed files : ")
for fileName in wrongGuessFiles:
    print(fileName)
        
print("----------------------------------")




if(allVerdictCorrect) :
    print("\nAll guesses are correct\n")
else :
    print("\nAll guesses are NOT correct!!!!!!!!!!!!!!!!!!!!!\n")


print(f"\nwrong guesses : {wrongGuesses}")
print(f"Accuracy : {1 - (wrongGuesses / len(imageFileNames))}")

