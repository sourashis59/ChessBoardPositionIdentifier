# to be able to import modules from this directory, we need to add the path to sys.path
# so that python can search that path from the imported modules
import sys
import time
sys.path.append("./scripts/modules")



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
from modules.imageToFen import ImageToFenConverter


#* scripts/main/test/image1.png

# CHESS_ENGINE_PATH = "scripts/main/chessEngineBin/stockfish_15_x64_avx2.exe"
CHESS_ENGINE_PATH = "scripts/main/chessEngineBin/bluebetafish_64bit_windows.exe"


emptyNonemptySquareClassifier = load_model("scripts/main/trainedModels/emptyNonemptySquareClassifier/CNN/trainedModelCNN.h5" ) 
blackOrWhitePieceClassifier = load_model("scripts/main/trainedModels/blackOrWhitePieceClassifier/CNN/trainedModelCNN.h5" ) 
pieceTypeClassifier = load_model("scripts/main/trainedModels/pieceTypeClassifier/NewTrainingData/trainedModelCNN.h5" )




chessEngine = ChessEngineWrapper(CHESS_ENGINE_PATH)
imageToFenConverter = ImageToFenConverter(emptyNonemptySquareClassifier, blackOrWhitePieceClassifier, pieceTypeClassifier)

print("\n\nPress \"ctrl + c\" to terminate!!!!!!!\n\n\n")
while(True):
    sys.stdin.flush()
    sys.stdout.flush()
    
    # example path : "scripts/main/test/image8.png"
    imagePath = input("\n\n\n\nEnter image path : ")
    currentPlayer = input("Enter current player (w/b) : ")
    moveTime = input("Enter movetime(in ms) : ")

    startTime = time.process_time()
    fenString = imageToFenConverter.getFenFromImage(imagePath)  + " " + currentPlayer + " - - "
    endTime = time.process_time()


    print("\nFEN : " + fenString)   
    print(f"time taken : {endTime-startTime} seconds\n\n")


    chessEngine.setposition(fenString)
    # chessEngine.printPosition()

    print("\n\nPrinting analysis: ")
    print("-----------------------------------------------------------------------\n")
    bestMoveString = chessEngine.go(moveTime)





    # #*draw arrow on the given image
    # boardImage = cv2.imread(imagePath)
    # move = getCellFromAlgebricMove(bestMoveString)
    # drawArrowFromSourceToDestCell(sourceCell=move[0], destCell=move[1], image=boardImage)

    # # resize the image (faced some problem while showing the image in my laptop's display--> problem with display. so resizing the image)
    # boardImage = cv2.resize(boardImage, (500, 500)) 

    # #* display the image with arrow
    # cv2.imshow(imagePath, boardImage)
    # #* waitKey() waits for a key press to close the window and 0 specifies indefinite loop
    # cv2.waitKey(0)
    # #* cv2.destroyAllWindows() simply destroys all the windows we created.
    # cv2.destroyAllWindows()

# r1b2bkr/ppp3pp/2n5/3qp3/2B5/8/PPPP1PPP/RNB1K2R w KQkq - 0 1
# rnbq1bnr/pppkpppp/8/3p4/3P4/8/PPPKPPPP/RNBQ1BNR w - - 2 3






