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

from chessEngineWrapper import ChessEngineWrapper



sourceImagePath = "scripts/main/test/image8.png"

# CHESS_ENGINE_PATH = "scripts/main/chessEngineBin/stockfish_15_x64_avx2.exe"
CHESS_ENGINE_PATH = "scripts/main/chessEngineBin/bluebetafish_64bit_windows.exe"













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
            
        if (i < 7):
            result.append("/")

    
    return ("").join(result)









def getFenFromImagePath(imagePath):
    boardImage = Image.open(os.path.join(imagePath) )

    pieces = []
    board = []
    squares = getSquaresFromChessBoardImage(boardImage)

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





    fenString = boardToFENPieces(board)
    # print(f"\nfen = {fenString}" )
    # print(pieces)

    return fenString











# fenString = getFenFromImagePath(sourceImagePath)
# print("\n\n\nImage path : "  + sourceImagePath)
# print("Fen : " + fenString)




chessEngine = ChessEngineWrapper(CHESS_ENGINE_PATH)
print("\n\nPress \"ctrl + c\" to terminate!!!!!!!\n\n\n")
while(True):
    sys.stdin.flush()
    sys.stdout.flush()
    
    # example path : "scripts/main/test/image8.png"
    imagePath = input("\n\n\n\nEnter image path : ")
    currentPlayer = input("Enter current player (w/b) : ")
    moveTime = input("Enter movetime(in ms) : ")

    fenString = getFenFromImagePath(imagePath)  + " " + currentPlayer + " - - "
    print("\nFEN : " + fenString)

    chessEngine.setposition(fenString)
    # chessEngine.printPosition()

    print("\n\nPrinting analysis: ")
    print("-----------------------------------------------------------------------\n")

    chessEngine.go(moveTime)

# r1b2bkr/ppp3pp/2n5/3qp3/2B5/8/PPPP1PPP/RNB1K2R w KQkq - 0 1
# rnbq1bnr/pppkpppp/8/3p4/3P4/8/PPPKPPPP/RNBQ1BNR w - - 2 3
