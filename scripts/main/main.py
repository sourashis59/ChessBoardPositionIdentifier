# to be able to import modules from this directory, we need to add the path to sys.path
# so that python can search that path from the imported modules
import sys
sys.path.append("./scripts/modules")


from util import getSquaresFromChessBoardImage

import os
from PIL import Image
import pickle

from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgba2rgb
import numpy as np





emptyNonemptySquareClassifier = pickle.load(open("scripts/main/trainedModels/emptyNonemptySquareClassifier/trainedModel.p" , 'rb')) 
squareTypes = ['empty', 'non-empty']

blackOrWhitePieceClassifier = pickle.load(open("scripts/main/trainedModels/blackOrWhitePieceClassifier/trainedModel.p" , 'rb'))
pieceColors = ['white', 'black']

# pieceTypeClassifier = pickle.load(open("scripts/main/trainedModels/pieceTypeClassifier/trainedModel.p" , 'rb'))
pieceTypeClassifier = pickle.load(open("scripts/main/trainedModels/pieceTypeClassifier/onekBoroData/trainedModel.p" , 'rb'))
pieceTypes = ['king', 'queen', 'rook', 'bishop', 'knight', 'pawn']


sourceImagePath = "scripts/main/test/image2.png"






pieces = []
squares = getSquaresFromChessBoardImage(Image.open(os.path.join(sourceImagePath) ))

for i in range(len(squares)) :
    currRow = ""

    for j in range(len(squares[i])):
        data = []

        # convert from PIL.Image to skimage
        squares[i][j].save("scripts/main/temp/tempSquareImage.png")
        squareImage = imread("scripts/main/temp/tempSquareImage.png")

        

        squareImage = resize(squareImage, (20, 20))
        data.append(squareImage.flatten())
        data = np.asarray(data)

        # if image is of type RGBA (contains an extra channel : alpha channel), remove the Alpha channel
        squareImageForPieceTypeClassifier = None
        if(len(squareImage[0][0]) == 4):
            squareImageForPieceTypeClassifier = rgba2rgb(squareImage)


        # check if square contains any piece
            # check the color and type of the piece
        if( squareTypes[ (emptyNonemptySquareClassifier.predict( [data[0]] ))[0] ] == 'empty') :
            currRow += ( "|    " )
            # print("empty")
        else :
            currPieceColor = pieceColors[ (blackOrWhitePieceClassifier.predict([data[0]]))[0] ]
            currPieceType = pieceTypes[ (pieceTypeClassifier.predict( [squareImageForPieceTypeClassifier.flatten()] ) )[0] ]

            
            currPieceString = f"{currPieceColor[0]}"
            if(currPieceType == 'knight') :
                currPieceString += ('n')     
            else :
                currPieceString += (currPieceType[0])

            currRow += ( "| " + currPieceString + " ")        

            # print(currPieceColor + " " + currPieceType)
    
    currRow += '|'
    pieces.append(currRow  )





print("\nBoard : \n")

# print the board
print("-----------------------------------------")
for i in range(len(pieces)) :
    print(pieces[i])
    print("-----------------------------------------")

print("\n\n--------------------------------------------------\n")

# print(pieces)

