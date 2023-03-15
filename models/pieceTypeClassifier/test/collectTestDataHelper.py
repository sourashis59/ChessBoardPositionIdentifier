# ======================================OBJECTIVE====================================
#
#   get squares from given sourceImagePath and store them in destImagePath 
#
# ===================================================================================



# to be able to import modules from this directory, we need to add the path to sys.path
# so that python can search that path from the imported modules
import sys
sys.path.append("./scripts/modules")

from util import getSquaresFromChessBoardImage
import os
from PIL import Image





sourceImagePath = "data/rawData/Lichess/piecetype12/Screenshot 2023-03-12 at 14-45-15 Chess analysis board.png"
imageCounter = 6

destPath = "models/pieceTypeClassifier/data/testData"


currImage = Image.open( os.path.join(sourceImagePath) )

squares = getSquaresFromChessBoardImage(currImage)
for i in range(0, len(squares)):
    for j in range(0, len(squares[i])):
        
        resizedImage = squares[i][j]


        # imageCounter = fileName[9:]
        fileName = f'{imageCounter}_{i}_{j}.png'

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
        if( (i == 0 and j == 0) ):
            pieceType = 'rook'
            pieceColor = 'black'
        elif( (i == 7 and j == 0) ):
            pieceType = 'rook'
            pieceColor = 'white'

        # bishop
        if( (i == 0 and j == 2) ):
            pieceType = 'bishop'
            pieceColor = 'black'
        elif( (i == 7 and j == 2)  ):
            pieceType = 'bishop'
            pieceColor = 'white'

        # knight
        if( (i == 0 and j == 1) ):
            pieceType = 'knight'
            pieceColor = 'black'
        elif( (i == 7 and j == 1)  ):
            pieceType = 'knight'
            pieceColor = 'white'

        # pawn
        if( (i == 1 and j == 0)  ):
            pieceType = 'pawn'
            pieceColor = 'black'
        elif( (i == 6 and j == 0)  ):
            pieceType = 'pawn'
            pieceColor = 'white'

        if(pieceType != 'NULL') :
            resizedImage.save(f"{destPath}/{pieceColor}_{pieceType}_{fileName}")

       




print("\nCopying Completed !\n")




