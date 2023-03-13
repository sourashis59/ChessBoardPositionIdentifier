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





sourceImagePath = "data/rawData/Lichess/piecetype11/Screenshot 2023-03-12 at 14-41-50 Chess analysis board.png"
pieceTypeCounter = 205

destImagePath = "models/blackOrWhitePieceClassifier/data/testData"


currImage = Image.open( os.path.join(sourceImagePath) )

squares = getSquaresFromChessBoardImage(currImage)
for i in range(0, len(squares)):
    for j in range(0, len(squares[i])):
        
        resizedImage = squares[i][j]


        # pieceTypeCounter = fileName[9:]
        fileName = f'{pieceTypeCounter}_{i}_{j}.png'

        if(i == 0 or i == 1) :
            fileName = "black_" + fileName
        elif (i == 6 or i == 7) :
            fileName = "white_" + fileName    

        if(i <= 1 or i >= 6):
            resizedImage.save(f"{destImagePath}/{fileName}")




print("\nCopying Completed !\n")




