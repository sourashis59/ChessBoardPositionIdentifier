
# from PIL import Image
import tensorflow as tf
import os 
import cv2
# import imghdr
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import Precision, Recall , BinaryAccuracy, SparseCategoricalAccuracy, Accuracy





resizedImageDimension = (64, 64)
# resizedImageDimension = (50, 50)







filePath = 'models/pieceTypeClassifier/data/testData/tempSquareImage_0_3.png'


# classifier = pickle.load(open('models/pieceTypeClassifier/trainedModel.p', 'rb'))
classifier = load_model( os.path.join("models/pieceTypeClassifier/trainedModelCNN.h5") )
classes = ['bishop', 'king', 'knight', 'pawn', 'queen', 'rook']


#* given a batch of input and model, returns the batch of predictions 
#* For example: given [image1, image2, ...., imageN]
#* Output : [predClass1, predClass2, ...., predClassN]
def getPredictions(batchX, model):

    #* it will return batch of predicted probabilities of classes 
    # *verbose = 0 ====> dont print anything while doing prediction
    pred = model.predict(batchX, verbose=0) 

    predClassIndices = []
    for i in range(len(pred)):
        predClassIndices.append(np.argmax(pred[i]))

    return predClassIndices        



print("\n\n\nCollecting data......................")



    
data = []
# images = []
imageFileNames = []



image = cv2.imread(filePath)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = tf.image.resize(image, resizedImageDimension).numpy().astype(int)
data.append(image)

plt.imshow(image)    
plt.show()

data = np.array(data)
#* IMPORTANT: we need to scale the pixel values from [0,255] to [0,1]
data = data / 255


print("Data Collected!\n")





#* for the CNN model, we need batch of images. 
# transformedData = np.expand_dims(data[i], 0)
transformedData = np.array([ data[0] ])

# #* it will return batch(of size 1) of predicted probabilities of classes(2 classes here)
# #* for example : pred = [[9.999664e-01 3.858412e-06]] 
# # *verbose = 0 ====> dont print anything while doing prediction
# pred = classifier.predict(transformedData, verbose=0) 
# # print(pred)
# predClassInd = np.argmax(pred[0])    
predClassIndices = getPredictions(transformedData, classifier)
predClassInd = predClassIndices[0]


print(f"\n\nPredicted class : {classes[predClassInd]}")