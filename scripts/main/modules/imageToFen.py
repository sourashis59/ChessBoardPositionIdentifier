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
from matplotlib import pyplot as plt
from PIL import Image
from tensorflow.keras.models import Sequential, load_model

from modules.util import *
from modules.chessEngineWrapper import ChessEngineWrapper




squareTypes =  ['emptySquare', 'nonemptySquare']
pieceColors = ['white', 'black']
pieceTypes = ['bishop', 'king', 'knight', 'pawn', 'queen', 'rook']



class ImageToFenConverter:
    
    #* private attributes
    __emptyNonemptySquareClassifier = None
    __blackOrWhitePieceClassifier = None
    __pieceTypeClassifier = None

    # __pieces = np.zeros((8,8))
    # __board = np.zeros((8,8))


    def __init__(self, emptyNonemptySquareClassifier, blackOrWhitePieceClassifier, pieceTypeClassifier):
        self.__emptyNonemptySquareClassifier = emptyNonemptySquareClassifier
        self.__blackOrWhitePieceClassifier = blackOrWhitePieceClassifier
        self.__pieceTypeClassifier = pieceTypeClassifier


    #
    # * @param pieces squares array from Board object (8*8 matrix)
    # * @returns string representing FEN field for piece position or null
    #
    def __boardToFENPieces(self, pieces) :
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


    def getFenFromImage(self, boardImagePath):
        boardImage = Image.open(os.path.join(boardImagePath) )
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

                # *convert PIL to openCV image
                # squareImageOpenCV = cv2.cvtColor(np.array(squares[i][j]), cv2.COLOR_RGB2BGR)
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
                if( squareTypes[ np.argmax( self.__emptyNonemptySquareClassifier.predict(transformedDataOpenCV, verbose=0)[0] ) ] == 'emptySquare') :
                    currRow += ( "|    " )
                    currRowBoard.append("NULL")
                    # print("empty")
                else :
                    currPieceColor = pieceColors[ (self.__blackOrWhitePieceClassifier.predict([data[0]]))[0] ]
                    currPieceType = pieceTypes[ np.argmax( self.__pieceTypeClassifier.predict(transformedDataOpenCV, verbose=0)[0] )  ]

                    
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


        fenString = self.__boardToFENPieces(board)
        # print(f"\nfen = {fenString}" )
        # print(pieces)

        return fenString






