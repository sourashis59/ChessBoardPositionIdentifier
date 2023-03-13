# ======================================OBJECTIVE====================================
#
# using  square images (with or without pieces) in "data/trainingData/emptyNonemptySquares/emptySquares" 
# and "data/trainingData/emptyNonemptySquares/nonemptySquares", train a model to detect if a given square image 
# is empty or it contains a piece  
#
# Store the trained model in "trainedModels/emptyNonemptySquareClassifier/model.p"
#
# ===================================================================================





from PIL import Image
from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2gray
import os
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
import pickle
import time




emptySquareSourcePath =    "data/trainingData/emptyNonemptySquares/emptySquares"
nonemptySquareSourcePath = "data/trainingData/emptyNonemptySquares/nonemptySquares"
trainedModelDestPath = "trainedModel/emptyNonemptySquareClassifier/model.p"


resizedImageDimension = (20, 20)



# ============================================================================================
# ====================================== PREPARE DATA ========================================
# ============================================================================================
print("Preparing data.......")
startTime = time.process_time()

data = []
labels = []

squareTypes = ['empty', 'non-empty']
for squareTypeInd, squareType in enumerate(squareTypes) :
    
    filePath = ""
    if(squareType == 'empty'):
        filePath = emptySquareSourcePath
    else:
        filePath = nonemptySquareSourcePath 

    for file in os.listdir(filePath):
        imagePath = os.path.join(filePath, file)
        # image = Image.open(imagePath)
        # image = image.resize((20,20))
        # data.append( list(image.getdata()) )
        
        image = imread(imagePath)
        
        # # make image black and white
        # # Only convert image to grayscale if it is RGB
        # if image.shape[-1] == 3:
        #     image = rgb2gray(image)

        image = resize(image, resizedImageDimension)
        data.append(image.flatten())
        
        
        labels.append(squareTypeInd)



data = np.asarray(data)
labels = np.asarray(labels)


# nsamples, nx, ny = data.shape
# data = data.reshape((nsamples,nx*ny))


# pickle.dump(data, open("Board_Extractor/TrainedModel/emptyNonemptySquareModelData.p", 'wb'))
# pickle.dump(labels, open("Board_Extractor/TrainedModel/emptyNonemptySquareModelLabels.p", 'wb'))


endTime = time.process_time()
print("Data Prepared!")
print("Elapsed time : ", (endTime - startTime) , " seconds.")
print() # for new line










# ============================================================================================
# ====================================== TRAIN TEST SPLIT ====================================
# ============================================================================================

print("Doing train_test_split........")
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.4, shuffle=True, stratify=labels)
print("Completed train_test_split!\n")









# ============================================================================================
# ====================================== TRAIN THE MODEL =====================================
# ============================================================================================
print("training the model......")
print(f"data size = {len(data)}, no. of features = {resizedImageDimension[0] * resizedImageDimension[1]}")
startTime = time.process_time()

classifier = SVC()
# parameters = [{'gamma' : [0.01, 0.001, 0.0001], 'C' : [1, 10, 100, 1000] }]
parameters = [{'gamma' : [0.01], 'C' : [1] }]

gridSearch = GridSearchCV(classifier, parameters)
gridSearch.fit(x_train, y_train)


endTime = time.process_time()
print("model  trained!")
print("Elapsed time : ", (endTime - startTime) , " seconds.")
print()




# ============================================================================================
# ====================================== TEST PERFORMANCE =====================================
# ============================================================================================
print("\nTESTING PERFORMANCE........")
startTime = time.process_time()

bestClassifier = gridSearch.best_estimator_
y_pred = bestClassifier.predict(x_test)
score = accuracy_score(y_pred, y_test)
print(f"Accuracy = {score*100}%" )

endTime = time.process_time()
print("Elapsed time : ", (endTime - startTime) , " seconds.")
print()



# ============================================================================================
# ====================================== SAVE THE MODEL =====================================
# ============================================================================================
print("Saving model")
pickle.dump(bestClassifier, open(trainedModelDestPath, 'wb'))
print("Model Saved\n")

