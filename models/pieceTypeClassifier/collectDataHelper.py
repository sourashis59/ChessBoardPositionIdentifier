# ======================================OBJECTIVE====================================
# for all board images in "data/trainingData/allImages", extract the pieces and
# save them in "models/pieceTypeClassifier/data/trainingData/images/{PIECE_TYPE}"
# ===================================================================================





# to be able to import modules from this directory, we need to add the path to sys.path
# so that python can search that path from the imported modules
import sys
sys.path.append("./scripts/modules")


from scripts.main.modules.util import getSquaresFromChessBoardImage

import os
from PIL import Image





pieceTypes = ['king', 'queen', 'rook', 'bishop', 'knight', 'pawn']


sourcePath = "data/trainingData/allImages"
destPath = "models/pieceTypeClassifier/data/trainingData/images"


print("Copying images from \'" + sourcePath + "\' ........")


sourceDirectory = os.fsencode(sourcePath)
imageCount = 0
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

                fileName = f'{imageCount}_{i}_{j}.png'



                pieceType = 'NULL'
                pieceColor = 'NULL'

                # king
                if( i == 0 and j == 4 ):
                    pieceType = 'king'
                    pieceColor = 'black'
                elif( i == 7 and j == 4 ):
                    pieceType = 'king'
                    pieceColor = 'white'

                # queen
                if( i == 0 and j == 3 ):
                    pieceType = 'queen'
                    pieceColor = 'black'
                elif( i == 7 and j == 3 ):
                    pieceType = 'queen'
                    pieceColor = 'white'

                # rook
                if( (i == 0 and j == 0) or (i == 0 and j == 7) ):
                    pieceType = 'rook'
                    pieceColor = 'black'
                elif( (i == 7 and j == 0) or (i == 7 and j == 7) ):
                    pieceType = 'rook'
                    pieceColor = 'white'

                # bishop
                if( (i == 0 and j == 2) or (i == 0 and j == 5) ):
                    pieceType = 'bishop'
                    pieceColor = 'black'
                elif( (i == 7 and j == 2) or (i == 7 and j == 5) ):
                    pieceType = 'bishop'
                    pieceColor = 'white'

                # knight
                if( (i == 0 and j == 1) or (i == 0 and j == 6) ):
                    pieceType = 'knight'
                    pieceColor = 'black'
                elif( (i == 7 and j == 1) or (i == 7 and j == 6) ):
                    pieceType = 'knight'
                    pieceColor = 'white'

                # pawn
                if( (i == 1 and j == 0) or (i == 1 and j == 1) ):
                    pieceType = 'pawn'
                    pieceColor = 'black'
                elif( (i == 6 and j == 0) or (i == 6 and j == 1) ):
                    pieceType = 'pawn'
                    pieceColor = 'white'

                if(pieceType != 'NULL') :
                    resizedImage.save(f"{destPath}/{pieceType}/{pieceColor}_{pieceType}_{fileName}")
           
                



             

    imageCount = imageCount + 1



print("\nCopying Completed !\n")




