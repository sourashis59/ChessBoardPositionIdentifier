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




classifier = pickle.load(open('models/blackOrWhitePieceClassifier/trainedModel.p', 'rb'))

pieceColor = ['white', 'black']

data = []
images = []
imageFileNames = []

filePath = 'models/blackOrWhitePieceClassifier/data/testData'

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


allVerdictCorrect = True
for i in range(len(data)):
    # images[i].show()

    pred = classifier.predict([data[i]])
    verdict = "NULL"
    if( imageFileNames[i].startswith(pieceColor[pred[0]]) ) :
        verdict = "correct" 
    else :
        verdict = "wrong"    
        allVerdictCorrect = False

    print(f"i = {i}, { pieceColor[pred[0]] },  verdict = {verdict}")    
    

if(allVerdictCorrect) :
    print("\nAll guesses are correct\n")
else :
    print("\nAll guesses are NOT correct!!!!!!!!!!!!!!!!!!!!!\n")
        
        
print("----------------------------------")
