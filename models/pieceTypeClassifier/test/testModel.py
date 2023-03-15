# ======================================OBJECTIVE====================================
#
# predict output for data using the models in "trainedModel.p"
# 
# ===================================================================================



from PIL import Image
from skimage.io import imread
from skimage.transform import resize
import os
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.svm import SVC
import pickle




classifier = pickle.load(open('models/pieceTypeClassifier/trainedModel.p', 'rb'))

pieceTypes = ['king', 'queen', 'rook', 'bishop', 'knight', 'pawn']

data = []
images = []
imageFileNames = []

filePath = 'models/pieceTypeClassifier/data/testData'

for file in os.listdir(filePath):
    imagePath = os.path.join(filePath, file)
    imageFileNames.append(file)

    # image = Image.open(imagePath)
    # images.append(image)
    # image = image.resize((20,20))
    # data.append( list(image.getdata()) )


    image = imread(imagePath)
    images.append(Image.open(imagePath))

    image = resize(image, (20, 20))
    data.append(image.flatten())



data = np.asarray(data)

# nsamples, nx, ny = data.shape
# data = data.reshape((nsamples,nx*ny))



print(f"images len = {len(images)}")
# images[40].show()


wrongVerdictCount = 0 
for i in range(len(data)):
    # images[i].show()

    pred = classifier.predict([data[i]])
    verdict = "NULL"
    
    # if imageFileNames[i] contains pieceTypes[pred[0]] as substring
    if( pieceTypes[pred[0]] in imageFileNames[i]  ) :
        verdict = "correct" 
    else :
        images[i].show()

        verdict = "wrong"    
        wrongVerdictCount = wrongVerdictCount + 1


    actualPieceType = 'NULL'
    if('king' in imageFileNames[i]) :
        actualPieceType = 'king'
    if('queen' in imageFileNames[i]) :
        actualPieceType = 'queen'
    if('rook' in imageFileNames[i]) :
        actualPieceType = 'rook'
    if('bishop' in imageFileNames[i]) :
        actualPieceType = 'bishop'
    if('knight' in imageFileNames[i]) :
        actualPieceType = 'knight'
    if('pawn' in imageFileNames[i]) :
        actualPieceType = 'pawn'
        

    print(f"i = {i}, ActualPieceType = { actualPieceType}, GuessedPieceType = {pieceTypes[pred[0]]},  verdict = {verdict}")    
    

if(wrongVerdictCount == 0) :
    print("\nAll guesses are correct\n")
else :
    print("\nAll guesses are NOT correct!!!!!!!!!!!!!!!!!!!!!\n")
    # print(f"Accuracy : {(1 - wrongVerdictCount/len(data)) * 100}")
    print(f"wrong verdict count = {wrongVerdictCount}")    

print("----------------------------------\n")
