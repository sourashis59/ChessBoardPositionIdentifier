
# to be able to import modules from this directory, we need to add the path to sys.path
# so that python can search that path from the imported modules
import sys
sys.path.append("./scripts/modules")




from PIL import Image
import os
from util import getSquaresFromChessBoardImage, getSquaresFromFenString







# collabSourcePath = "/content/drive/MyDrive/ChessBoardStateExtractor"

destPath = "models/emptyNonemptySquareClassifier/data/testData"
lichessChessdotcomSourcePath = "data/rawData/lichess_chess.com_images"
kaggleDataSourcePath = "data/rawData/kaggleData/train"




# =========================================================lichessChessdotcom data=========================================================
print("Copying images from \'" + lichessChessdotcomSourcePath + "\' ........")



sourceDirectory = os.fsencode(lichessChessdotcomSourcePath)
imageCount = 0
lichessChessdotcomImageCount = 0
maxLichessChessdotcomImageCount = 10

for file in os.listdir(sourceDirectory):
    fileName = os.fsdecode(file)
    if ( fileName.endswith(".jpg") or fileName.endswith(".png") ) :                 
        # print(os.path.join(fileName))

        print(f"fileName : {fileName}")

        currImage = Image.open( os.path.join(sourceDirectory.decode(), fileName ) )

        squares = getSquaresFromChessBoardImage(currImage)
        for i in range(0, len(squares)):
            for j in range(0, len(squares[i])):
                
                # # resize the images to newWidth * newHeight resolution
                # newWidth = 100
                # newHeight = 100
                # resizedImage = squares[i][j].resize((newWidth, newHeight))
                
                resizedImage = squares[i][j]


                # only save the pieces on first 5 columns of 0-th and last row
                # and 2 pawns for each color
                # and 2 rows of empty squares
                if ((i == 0 or i == 7) and j <= 4) or ((i == 1 or i == 6) and j <= 1)  :
                    resizedImage.save(f"{destPath}/nonemptySquare_{imageCount}.png")
                    imageCount = imageCount + 1
                elif  (i == 2 or i == 3):
                    resizedImage.save(f"{destPath}/emptySquare_{imageCount}.png")
                    imageCount = imageCount + 1


    lichessChessdotcomImageCount += 1

    # we will use maxlichessChessdotcomImageCount kaggle data
    if(lichessChessdotcomImageCount > maxLichessChessdotcomImageCount) :
        break


    # print("fen = "+ fenString)
    if(lichessChessdotcomImageCount % 100 == 0) :
        print(f"processed images = {lichessChessdotcomImageCount}/{maxLichessChessdotcomImageCount}")



print("\nCopying Completed !\n")

print(f"lichessChessdotcomImageCount = {lichessChessdotcomImageCount}")
print(f"saved imageCount = {imageCount}\n\n\n\n")



















# =========================================================extra kaggle data=========================================================



# we will get maxKaggleDataCount data from kaggle data 
kaggleDataCount = 0
maxKaggleDataCount = 10

sourceDirectory = os.fsencode(kaggleDataSourcePath)
for file in os.listdir(sourceDirectory):
    fileName = os.fsdecode(file)

    # print("filename : " + fileName)
    
    if ( fileName.endswith(".jpg") or fileName.endswith(".png") or fileName.endswith(".jpeg") ) :                 
        # print(os.path.join(fileName))
        
        fenString = fileName[0: (fileName.index('.') ) ].replace('-', '/') 
        
        

        squares = getSquaresFromFenString(fenString)

        squareImages = getSquaresFromChessBoardImage(Image.open(os.path.join(kaggleDataSourcePath, fileName)))    

        emptySquareImages = []
        nonemptySquareImages = []
        for i in range(8) :
            for j in range(8) :

                pieceType = "NULL"
                color = "NULL"
                if(squares[i][j] == 'k'):
                    pieceType = "king"
                    color = 'black'
                elif(squares[i][j] == 'K'):
                    pieceType = "king"
                    color = 'white'
                
                elif(squares[i][j] == 'q'):
                    pieceType = "queen"
                    color = 'black'
                elif(squares[i][j] == 'Q'):
                    pieceType = "queen"
                    color = 'white'
                
                elif(squares[i][j] == 'r'):
                    pieceType = "rook"
                    color = 'black'
                elif(squares[i][j] == 'R'):
                    pieceType = "rook"
                    color = 'white'
                    
                elif(squares[i][j] == 'b'):
                    pieceType = "bishop"
                    color = 'black'
                elif(squares[i][j] == 'B'):
                    pieceType = "bishop"
                    color = 'white'

                elif(squares[i][j] == 'n'):
                    pieceType = "knight"
                    color = 'black'
                elif(squares[i][j] == 'N'):
                    pieceType = "knight"
                    color = 'white'    

                elif(squares[i][j] == 'p'):
                    pieceType = "pawn"
                    color = 'black'
                elif(squares[i][j] == 'P'):
                    pieceType = "pawn"
                    color = 'white'    



                # # save the image
                # if(pieceType != "NULL") :
                #     squareImages[i][j].save(os.path.join(destPath, f"{pieceType}/{color}/{imageCount}_{color}_{pieceType}.png"))
                #     imageCount += 1
        


                # we will store equal amount of nonempty and empty squares
                if(pieceType != "NULL") :
                    nonemptySquareImages.append(squareImages[i][j])
                else :
                    emptySquareImages.append(squareImages[i][j])
            

        emptySquareImages = emptySquareImages[:min(len(nonemptySquareImages), len(emptySquareImages))]
        for image in emptySquareImages:
            image.save(f"{destPath}/emptySquare_kaggleData_{imageCount}.png")
            imageCount = imageCount + 1

        for image in nonemptySquareImages:
            image.save(f"{destPath}/nonemptySquare_kaggleData_{imageCount}.png")
            imageCount = imageCount + 1



        kaggleDataCount += 1

        # we will use maxKaggleDataCount kaggle data
        if(kaggleDataCount > maxKaggleDataCount) :
            break


        # print("fen = "+ fenString)
        if(kaggleDataCount % 100 == 0) :
          print(f"processed images = {kaggleDataCount}/{maxKaggleDataCount}")

                                    

print("\nCopying Completed !\n")

print(f"imageCOunt = {imageCount}" )
