# ======================================OBJECTIVE====================================
# for all board images in "data/trainingData/allImages", extract the squares with the pieces
# and 2 rows of empty squares
# save them in "models/emptyNonemptySquareClassifier/data/trainingData/images/emptySquares"
# and "models/emptyNonemptySquareClassifier/data/trainingData/images/nonemptySquares"
# ===================================================================================





# to be able to import modules from this directory, we need to add the path to sys.path
# so that python can search that path from the imported modules
import sys
sys.path.append("./scripts/modules")


from util import getSquaresFromChessBoardImage

import os
from PIL import Image






sourcePath = "data/trainingData/allImages"
emptySquareDestPath = "models/emptyNonemptySquareClassifier/data/trainingData/images/emptySquares"
nonemptySquareDestPath = "models/emptyNonemptySquareClassifier/data/trainingData/images/nonemptySquares"


print("Copying images from \'" + sourcePath + "\' ........")


sourceDirectory = os.fsencode(sourcePath)
pieceSetCount = 0
for file in os.listdir(sourceDirectory):
    fileName = os.fsdecode(file)
    if ( fileName.endswith(".jpg") or fileName.endswith(".png") ) :                 
        # print(os.path.join(fileName))
        currImage = Image.open( os.path.join(sourceDirectory.decode(), fileName ) )

        squares = getSquaresFromChessBoardImage(currImage)
        for i in range(0, len(squares)):
            for j in range(0, len(squares[i])):
                
                # # resize the images to newWidth * newHeight resolution
                # newWidth = 100
                # newHeight = 100
                # resizedImage = squares[i][j].resize((newWidth, newHeight))
                
                resizedImage = squares[i][j]

                fileName = f'{pieceSetCount}_{i}_{j}.png'

                # only save the pieces on first 5 columns of 0-th and last row
                # and 2 pawns for each color
                # and 2 rows of empty squares
                if ((i == 0 or i == 7) and j <= 4) or ((i == 1 or i == 6) and j <= 1)  :
                    resizedImage.save(f"{nonemptySquareDestPath}/{fileName}")
                elif  (i == 2 or i == 3):
                    resizedImage.save(f"{emptySquareDestPath}/{fileName}")

    pieceSetCount = pieceSetCount + 1



print("\nCopying Completed !\n")




