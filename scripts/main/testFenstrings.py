# to be able to import modules from this directory, we need to add the path to sys.path
# so that python can search that path from the imported modules
import sys
sys.path.append("./scripts/modules")


from scripts.main.modules.util import getSquaresFromChessBoardImage

import os
from PIL import Image
import pickle

from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgba2rgb
import numpy as np

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




sourceDirPath = "scripts/main/fenstringImages"








# emptyNonemptySquareClassifier = pickle.load(open("scripts/main/trainedModels/emptyNonemptySquareClassifier/21-03-2023/trainedModel.p" , 'rb')) 
emptyNonemptySquareClassifier = load_model("scripts/main/trainedModels/emptyNonemptySquareClassifier/21-03-2023/trainedModelCNN.h5" ) 
squareTypes =  ['emptySquare', 'nonemptySquare']

blackOrWhitePieceClassifier = pickle.load(open("scripts/main/trainedModels/blackOrWhitePieceClassifier/trainedModel.p" , 'rb'))
pieceColors = ['white', 'black']

# pieceTypeClassifier = pickle.load(open("scripts/main/trainedModels/pieceTypeClassifier/trainedModel.p" , 'rb'))
# pieceTypes = ['king', 'queen', 'rook', 'bishop', 'knight', 'pawn']
pieceTypeClassifier = load_model("scripts/main/trainedModels/pieceTypeClassifier/22-03-2023-sparse_categorical_accuracy/trainedModelCNN.h5" )
pieceTypes = ['bishop', 'king', 'knight', 'pawn', 'queen', 'rook']







#
# * @param pieces squares array from Board object (8*8 matrix)
# * @returns string representing FEN field for piece position or null
#
def boardToFENPieces(pieces) :
    result = []
    
    for i in range(8):
        for  j in range(8):
            if (pieces[i][j] != "NULL" ) :
                result.append(pieces[i][j])
            elif (len(result) == 0 or not result[len(result) - 1].isdigit()) :
                result.append("1")
            else :
                if (int(result[len(result) - 1]) > 8):
                    return "NULL"
                result[len(result) - 1] = str( int(result[len(result) - 1]) + 1 )
            
        if (i < 8 - 1):
            result.append("/")

    
    return ("").join(result)


















wrongGuessCount = 0
totalFileCount = 0
wrongGuesses = []

for file in os.listdir(sourceDirPath):
    totalFileCount += 1

    pieces = []
    board = []
    squares = getSquaresFromChessBoardImage(Image.open(os.path.join(sourceDirPath, file)))

    for i in range(len(squares)) :
        currRow = ""
        currRowBoard = []

        for j in range(len(squares[i])):
            data = []

            # convert from PIL.Image to skimage
            squares[i][j].save("scripts/main/temp/tempSquareImage.png")
            squareImage = imread("scripts/main/temp/tempSquareImage.png")

            squareImageOpenCV = cv2.imread("scripts/main/temp/tempSquareImage.png")
            squareImageOpenCV = cv2.cvtColor(squareImageOpenCV, cv2.COLOR_BGR2RGB)
            squareImageOpenCV = tf.image.resize(squareImageOpenCV, (64, 64)).numpy().astype(int)
            transformedDataOpenCV = np.expand_dims(squareImageOpenCV/255, 0)

            squareImage = resize(squareImage, (20, 20))
            # if image is of type RGBA (contains an extra channel : alpha channel), remove the Alpha channel
            if(len(squareImage[0][0]) == 4):
                squareImage = rgba2rgb(squareImage)

            data.append(squareImage.flatten())
            data = np.asarray(data)

            


            # check if square contains any piece
                # check the color and type of the piece
            if( squareTypes[ np.argmax( emptyNonemptySquareClassifier.predict(transformedDataOpenCV, verbose=0)[0] ) ] == 'emptySquare') :
                currRow += ( "|    " )
                currRowBoard.append("NULL")
                # print("empty")
            else :
                currPieceColor = pieceColors[ (blackOrWhitePieceClassifier.predict([data[0]]))[0] ]
                currPieceType = pieceTypes[ np.argmax( pieceTypeClassifier.predict(transformedDataOpenCV, verbose=0)[0] )  ]

                
                currPieceString = f"{currPieceColor[0]}"
                if(currPieceType == 'knight') :
                    currPieceString += ('n')     
                    currRowBoard.append('n')
                else :
                    currPieceString += (currPieceType[0])
                    currRowBoard.append(currPieceType[0])

                if(currPieceColor == "white"):
                    currRowBoard[len(currRowBoard) - 1] = currRowBoard[len(currRowBoard) - 1].upper() 

                currRow += ( "| " + currPieceString + " ")        

                # print(currPieceColor + " " + currPieceType)
        
        currRow += '|'
        pieces.append(currRow  )
        board.append(currRowBoard)




    # print("\nPieces : \n")

    # # print the board
    # print("-----------------------------------------")
    # for i in range(len(pieces)) :
    #     print(pieces[i])
    #     print("-----------------------------------------")

    # print("\n\n--------------------------------------------------\n")



    guessedFenString = boardToFENPieces(board)
    # print(f"\nfen = {guessedFenString}" )

    actualFenString = file[0: (file.index('.') ) ].replace('-', '/') 
    
    verdict = (guessedFenString == actualFenString)
    print(f"\nVerdict: {verdict}, fileNo. : {totalFileCount}, wrongGuessCount = {wrongGuessCount}")
    print(f"Actual Fen  : {actualFenString}")
    print(f"Guessed Fen : {guessedFenString}")

    if(verdict == False):
        wrongGuessCount += 1
        wrongGuesses.append(file)



print("\n\n\n\nWrongly guesses filenamse : ")
for file in wrongGuesses:
    print(file)

print(f"\n\n\nWrong guess count = {wrongGuessCount}")
print(f"Accuracy = {1 - (wrongGuessCount/totalFileCount)}")





# rnbqkbnr-pppppppp-8-8-8-8-PPPPPPPP-RNBQKBNR
