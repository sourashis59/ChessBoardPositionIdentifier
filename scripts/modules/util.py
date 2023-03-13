from PIL import Image
import os


# returns a 2d array of squares(images) from the given chees board image
def getSquaresFromChessBoardImage(chessBoardImage):
    
    imageWidth, imageHeight = chessBoardImage.size

    # print("ORIGINAL IMAGE : imageWidth = " , imageWidth , ", imageHeight = " , imageHeight)
    # box = (0,0, imageWidth , imageHeight)
    # croppedImage = image.crop(box)
    # print("CROPPED IMAGE :  imageWidth = " , croppedImage.size[0] , ", imageHeight = " , croppedImage.size[1])

    squares = []
    for i in range(0, 8):
        currRowOfSquares = []
        for j in range(0, 8):
            left = j * imageWidth / 8
            right = (j + 1) * imageWidth / 8
            upper = i * imageHeight / 8
            lower = (i + 1) * imageHeight / 8

            currBox = (left, upper, right, lower)
            currSquare = chessBoardImage.crop(currBox)

            currRowOfSquares.append(currSquare)


        squares.append(currRowOfSquares)


    # return the squares[]][]
    return squares
