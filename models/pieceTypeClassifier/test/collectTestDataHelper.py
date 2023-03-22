
# to be able to import modules from this directory, we need to add the path to sys.path
# so that python can search that path from the imported modules
import sys
sys.path.append("./scripts/modules")




from PIL import Image
import os
from util import getSquaresFromChessBoardImage, getSquaresFromFenString
import random






#* if true, dump the files in destPathForEvaulate inside subdirectories (king, queen, rook, bishop, ..... ) [to be used by tf.keras.models.evaluate()]
#* else dump all the fiels in destPath 
collectTestDataForEvaluate = True









destPath = "data/testData/pieceTypeClassifier"
destPathForEvaluate = "data/testData/pieceTypeClassifierForEvaluate"

lichessChessdotcomSourcePath = "data/rawData/lichess_chess.com_images"
kaggleDataSourcePath = "data/rawData/kaggleData/test"



maxLichessChessdotcomImageCount = 2000
maxKaggleDataCount = 0



classes = ['bishop', 'king', 'knight', 'pawn', 'queen', 'rook']


# *create the subdirectories if not created earlier
if(collectTestDataForEvaluate) :
    for subdir in classes:
        newPath = os.path.join(destPathForEvaluate, subdir)
        if not os.path.exists(newPath):
            os.makedirs(newPath)





# =========================================================lichessChessdotcom data=========================================================
print("Copying images from \'" + lichessChessdotcomSourcePath + "\' ........")



sourceDirectory = os.fsencode(lichessChessdotcomSourcePath)
imageCount = 0
lichessChessdotcomImageCount = 0


for file in os.listdir(sourceDirectory):
    fileName = os.fsdecode(file)
    if ( fileName.endswith(".jpg") or fileName.endswith(".png") ) :                 
        # print(os.path.join(fileName))

        # we will use maxlichessChessdotcomImageCount kaggle data
        if(lichessChessdotcomImageCount > maxLichessChessdotcomImageCount) :
            break
        # print("fen = "+ fenString)
        if(lichessChessdotcomImageCount % 100 == 0) :
            print(f"processed images = {lichessChessdotcomImageCount}/{maxLichessChessdotcomImageCount}")

        lichessChessdotcomImageCount += 1



        currImage = Image.open( os.path.join(sourceDirectory.decode(), fileName ) )
        squares = getSquaresFromChessBoardImage(currImage)
        for i in range(0, len(squares)):
            for j in range(0, len(squares[i])):
                
                # # resize the images to newWidth * newHeight resolution
                # newWidth = 100
                # newHeight = 100
                # resizedImage = squares[i][j].resize((newWidth, newHeight))
                
                resizedImage = squares[i][j]


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
                    if(collectTestDataForEvaluate) :
                        resizedImage.save(f"{destPathForEvaluate}/{pieceType}/{pieceType}_{pieceColor}_{imageCount}.png")
                    else:                        
                        resizedImage.save(f"{destPath}/{pieceType}_{pieceColor}_{imageCount}.png")

                    imageCount += 1








print("\nCopying Completed !\n")

print(f"lichessChessdotcomImageCount = {lichessChessdotcomImageCount}")
print(f"saved imageCount = {imageCount}\n\n\n\n")



















# =========================================================extra kaggle data=========================================================



# we will get maxKaggleDataCount data from kaggle data 
kaggleDataCount = 0

sourceDirectory = os.fsencode(kaggleDataSourcePath)
for file in os.listdir(sourceDirectory):
    fileName = os.fsdecode(file)

    # print("filename : " + fileName)
    
    if ( fileName.endswith(".jpg") or fileName.endswith(".png") or fileName.endswith(".jpeg") ) :                 
        # print(os.path.join(fileName))

        # we will use maxKaggleDataCount kaggle data
        if(kaggleDataCount > maxKaggleDataCount) :
            break

        # print("fen = "+ fenString)
        if(kaggleDataCount % 100 == 0) :
          print(f"processed images = {kaggleDataCount}/{maxKaggleDataCount}")

        kaggleDataCount += 1




        
        fenString = fileName[0: (fileName.index('.') ) ].replace('-', '/') 
        
        

        squares = getSquaresFromFenString(fenString)

        squareImages = getSquaresFromChessBoardImage(Image.open(os.path.join(kaggleDataSourcePath, fileName)))    

        emptySquareImages = []
        nonemptySquareImages = []
        for i in range(8) :
            for j in range(8) :

                pieceType = "NULL"
                pieceColor = "NULL"
                if(squares[i][j] == 'k'):
                    pieceType = "king"
                    pieceColor = 'black'
                elif(squares[i][j] == 'K'):
                    pieceType = "king"
                    pieceColor = 'white'
                
                elif(squares[i][j] == 'q'):
                    pieceType = "queen"
                    pieceColor = 'black'
                elif(squares[i][j] == 'Q'):
                    pieceType = "queen"
                    pieceColor = 'white'
                
                elif(squares[i][j] == 'r'):
                    pieceType = "rook"
                    pieceColor = 'black'
                elif(squares[i][j] == 'R'):
                    pieceType = "rook"
                    pieceColor = 'white'
                    
                elif(squares[i][j] == 'b'):
                    pieceType = "bishop"
                    pieceColor = 'black'
                elif(squares[i][j] == 'B'):
                    pieceType = "bishop"
                    pieceColor = 'white'

                elif(squares[i][j] == 'n'):
                    pieceType = "knight"
                    pieceColor = 'black'
                elif(squares[i][j] == 'N'):
                    pieceType = "knight"
                    pieceColor = 'white'    

                elif(squares[i][j] == 'p'):
                    pieceType = "pawn"
                    pieceColor = 'black'
                elif(squares[i][j] == 'P'):
                    pieceType = "pawn"
                    pieceColor = 'white'    


                if(pieceType != 'NULL') :
                    if(collectTestDataForEvaluate) :
                        squareImages[i][j].save(f"{destPathForEvaluate}/{pieceType}/{pieceType}_{pieceColor}_kaggleData_{imageCount}.png")
                    else:                        
                        squareImages[i][j].save(f"{destPath}/{pieceType}_{pieceColor}_kaggleData_{imageCount}.png")

                    imageCount += 1




        

                                    

print("\nCopying Completed !\n")

print(f"imageCOunt = {imageCount}" )
