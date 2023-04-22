

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
from modules.chessEngineWrapper import ChessEngineWrapper
from modules.imageToFen import ImageToFenConverter



# sourceDirPath = "scripts/main/fenstringImages"
# sourceDirPath = "data/rawData/kaggleData/test"
sourceDirPath = "data/rawData/lichess_chess.com_images"





emptyNonemptySquareClassifier = load_model("scripts/main/trainedModels/emptyNonemptySquareClassifier/CNN/trainedModelCNN.h5" ) 
blackOrWhitePieceClassifier = load_model("scripts/main/trainedModels/blackOrWhitePieceClassifier/CNN/trainedModelCNN.h5" ) 
pieceTypeClassifier = load_model("scripts/main/trainedModels/pieceTypeClassifier/NewTrainingData/trainedModelCNN.h5" )


imageToFenConverter = ImageToFenConverter(emptyNonemptySquareClassifier, blackOrWhitePieceClassifier, pieceTypeClassifier)

wrongGuessCount = 0
totalFileCount = 0
wrongGuesses = []

for file in os.listdir(sourceDirPath):
    totalFileCount += 1

    guessedFenString = imageToFenConverter.getFenFromImage(os.path.join(sourceDirPath, file))
    if("lichess" in sourceDirPath or "chess.com" in sourceDirPath):
        actualFenString = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
    else:
        actualFenString = file[0: (file.index('.') ) ].replace('-', '/') 
    
    verdict = (guessedFenString == actualFenString)
    # print(f"\nVerdict: {verdict}, fileNo. : {totalFileCount}, wrongGuessCount = {wrongGuessCount}")
    # print(f"Actual Fen  : {actualFenString}")
    # print(f"Guessed Fen : {guessedFenString}")

    if(verdict == False):
        wrongGuessCount += 1
        # wrongGuesses.append(file)

    if(totalFileCount % 100 == 0):
        print(f"total processed files : {totalFileCount}, wrongGuesses : {wrongGuessCount}")


# print("\n\n\n\nWrongly guesses filenamse : ")
# for file in wrongGuesses:
#     print(file)

print(f"\n\n\nWrong guess count = {wrongGuessCount}")
print(f"Accuracy = {1 - (wrongGuessCount/totalFileCount)}")





# rnbqkbnr-pppppppp-8-8-8-8-PPPPPPPP-RNBQKBNR
