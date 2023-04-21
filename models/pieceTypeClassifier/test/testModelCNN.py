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
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import Precision, Recall , BinaryAccuracy, SparseCategoricalAccuracy, Accuracy





use_tf_keras_models_evaluate = False
resizedImageDimension = (64, 64)
# resizedImageDimension = (50, 50)

#* test without prinitng the wrongly guessed file names
testWithoutDebugging = True







filePath = 'data/testData/pieceTypeClassifier'
filePathForEvaluate = 'data/testData/pieceTypeClassifierForEvaluate'


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

if (testWithoutDebugging):

    # * data contains batches of {image, label}
    # * image is resized to <resizedImageDimension> and converted to "RGB"
    data = tf.keras.utils.image_dataset_from_directory(filePathForEvaluate, image_size=resizedImageDimension, shuffle=False)
    print("Data prepared...............\n")
    classes = data.class_names

    #* scale the image pixel values from range [0, 255] to [0,1] ==> this range helps the Depp learning model learn faster
    # batch[0] = batch[0] / 255

    #* x is image, y is label
    data = data.map(lambda x,y : (x/255, y)) 


    print("\n\n\ntesting Performance..................\n")
    #* calculate accuracy in hand
    correctGuess = 0
    totalGuess = 0

    #* using inbuilt functions to calculate accuracy
    precision = Precision()
    recall = Recall()
    # accuracy = BinaryAccuracy()
    accuracy = Accuracy()
    sparseCategoricalAccuracy = SparseCategoricalAccuracy()

    print(f"\nlen(data) = {len(data)}\n\n")
    batchCount = 0
    for batch in data.as_numpy_iterator() :
        
        if(batchCount % 100 == 0):
            print(f"Tested data : {batchCount}/{len(data)},   wrong guess : {totalGuess-correctGuess}")

        batchCount += 1

        # * y will be an array of labels (for example: y = [1,4,0,0,2,0,1,3,3,1,0,5])
        # * model.predict(X) will return 32*6 array of {probability of class1, prob of class2, ......} (batch size = 32)
        # *     (for example : model.predict(x) = [ [0.2, 0.4,...], [0.5,0.1,.....], [1, 0,....], [0.3, 0.2,....] ])
        # * np.argmax(arr) returns the index of the max element
        X, y = batch

        #* array of array of probabilites of the classes
        y_pred = classifier.predict(X, verbose=False) 
        sparseCategoricalAccuracy.update_state(y, y_pred)

        #* array of label indices
        y_pred = [ np.argmax(element) for element in y_pred] #* for each probablity (size 6 array), find the index of max 

        totalGuess += len(y)
        for i in range(len(y)):
            if(y[i] == y_pred[i]):
                correctGuess += 1

        precision.update_state(y, y_pred)
        recall.update_state(y, y_pred)
        accuracy.update_state(y, y_pred)


    print(f"\n\nIn hand calculation : accuracy =  {correctGuess/totalGuess}")
    print(f"Using inbuilt Accuracy() to calculate accuracy: Accuracy = {accuracy.result().numpy()}")
    print(f"Using inbuilt SparseCategoricalAccuracy() to calculate accuracy: SparseCategoricalAccuracy = {sparseCategoricalAccuracy.result().numpy()}")
    print(f"\nprecision = {precision.result().numpy()}, Recall = {recall.result().numpy()},")




else:
        
    data = []
    # images = []
    imageFileNames = []


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

        data = np.array(data)
        #* IMPORTANT: we need to scale the pixel values from [0,255] to [0,1]
        data = data / 255

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
            
            #* for the CNN model, we need batch of images. 
            # transformedData = np.expand_dims(data[i], 0)
            transformedData = np.array([ data[i] ])
            # print(transformedData)
            
            # plt.imshow(transformedData[0])
            # plt.show()

            # # pred = classifier.predict([data[i]])

            # #* it will return batch(of size 1) of predicted probabilities of classes(2 classes here)
            # #* for example : pred = [[9.999664e-01 3.858412e-06]] 
            # # *verbose = 0 ====> dont print anything while doing prediction
            # pred = classifier.predict(transformedData, verbose=0) 
            # # print(pred)
            # predClassInd = np.argmax(pred[0])    
            predClassIndices = getPredictions(transformedData, classifier)
            predClassInd = predClassIndices[0]

            verdict = "NULL"

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



    # # else :
    #     testData = tf.keras.utils.image_dataset_from_directory(filePathForEvaluate, image_size=resizedImageDimension)
    #     testData = testData.map(lambda x,y : (x/255, y)) 

        
    #     testingResult = classifier.evaluate(testData)
    #     print("\n\nTesting result (using evaluate() ): ")
    #     print(testingResult)
    #     print("\n\n\n\n\n")
