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
pieceColors = ['black', 'white']
pieceTypes = ['bishop', 'king', 'knight', 'pawn', 'queen', 'rook']



class ImageToFenConverter:
    
    #* private attributes
    __emptyNonemptySquareClassifier = None
    __blackOrWhitePieceClassifier = None
    __pieceTypeClassifier = None

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
        board = []
        squares = getSquaresFromChessBoardImage(boardImage)

        for i in range(len(squares)) :
            currRowBoard = []
            for j in range(len(squares[i])):
 
                # *convert PIL to openCV image
                squareImageOpenCV = cv2.cvtColor(np.array(squares[i][j]), cv2.COLOR_RGB2BGR) #* this is to drop the alpha channel(if any) of the PIL image
                squareImageOpenCV = cv2.cvtColor(squareImageOpenCV, cv2.COLOR_BGR2RGB)
                squareImageOpenCV = tf.image.resize(squareImageOpenCV, (64, 64)).numpy().astype(int)
                transformedDataOpenCV = np.expand_dims(squareImageOpenCV/255, 0)

                #* check if square contains any piece
                if( squareTypes[ np.argmax( self.__emptyNonemptySquareClassifier.predict(transformedDataOpenCV, verbose=0)[0] ) ] == 'emptySquare') :
                    currRowBoard.append("NULL")
                else :
                    currPieceColor = pieceColors[ np.argmax( self.__blackOrWhitePieceClassifier.predict(transformedDataOpenCV, verbose=0)[0] ) ]
                    currPieceType = pieceTypes[ np.argmax( self.__pieceTypeClassifier.predict(transformedDataOpenCV, verbose=0)[0] )  ]

                    if(currPieceType == 'knight') :
                        currRowBoard.append('n')
                    else :
                        currRowBoard.append(currPieceType[0])

                    if(currPieceColor == "white"):
                        currRowBoard[len(currRowBoard) - 1] = currRowBoard[len(currRowBoard) - 1].upper() 

            board.append(currRowBoard)

        fenString = self.__boardToFENPieces(board)
        # print(f"\nfen = {fenString}" )
        # print(pieces)

        return fenString






