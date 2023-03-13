# ======================================OBJECTIVE====================================
# for all board images in "data/trainingData/allImages", extract the squares with the pieces
# save them in "models/blackOrWhitePieceClassifier/data/trainingData/images/whitePieces"
# and "models/blackOrWhitePieceClassifier/data/trainingData/images/blackPieces"
# ===================================================================================





# to be able to import modules from this directory, we need to add the path to sys.path
# so that python can search that path from the imported modules
import sys
sys.path.append("./scripts/modules")


from util import getSquaresFromChessBoardImage

import os
from PIL import Image






sourcePath = "data/trainingData/allImages"
whitePiecesDestPath = "models/blackOrWhitePieceClassifier/data/trainingData/images/whitePieces"
blackPiecesDestPath = "models/blackOrWhitePieceClassifier/data/trainingData/images/blackPieces"


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
                if( (i == 0 and j <= 4) or (i == 1 and j<=1) ) :
                    resizedImage.save(f"{blackPiecesDestPath}/black_{fileName}")
                elif ( (i == 7 and j <= 4) or (i == 6 and j<=1) ) :
                    resizedImage.save(f"{whitePiecesDestPath}/white_{fileName}")


             

    pieceSetCount = pieceSetCount + 1



print("\nCopying Completed !\n")




