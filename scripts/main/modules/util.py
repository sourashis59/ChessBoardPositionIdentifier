from PIL import Image
import os
from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgba2rgb
import cv2
import tensorflow as tf
import numpy as np


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





















   





#*      0   1   2   3   4   5   6   7
#*    -----------------------------------
#*   0|   |   |   |   |   |   |   |   |
#*    |---|---|---|---|---|---|---|---|
#*   1|   |   |   |   |   |   |   |   |
#*    |---|---|---|---|---|---|---|---|
#*   2|   |   |   |   |   |   |   |   |
#*    |---|---|---|---|---|---|---|---|
#*   3|   | S | D |   |   |   |   |   |
#*    |---|---|---|---|---|---|---|---|
#*   4|   |   |   |   |   |   |   |   |
#*    |---|---|---|---|---|---|---|---|
#*   5|   |   |   |   |   |   |   |   |
#*    |---|---|---|---|---|---|---|---|
#*   6|   |   |   |   |   |   |   |   |
#*    |---|---|---|---|---|---|---|---|
#*   7|   |   |   |   |   |   |   |   |
#*    |---|---|---|---|---|---|---|---|
#*
#*   
#*    S ==> SOURCE CELL (3,1)
#*    D ==> DEST CELL (3,2)
#*
#*
#*               ||||
#*               ||||
#*               ||||        
#*               ||||    
#*               ||||        DRAW ARROW FROM SOURCE TO DEST
#*               ||||
#*               ||||
#*               ||||
#*             --------
#*             \      /
#*              \    /
#*               \  /
#*                \/
#*
#*      0   1   2   3   4   5   6   7
#*    -----------------------------------
#*   0|   |   |   |   |   |   |   |   |
#*    |---|---|---|---|---|---|---|---|
#*   1|   |   |   |   |   |   |   |   |
#*    |---|---|---|---|---|---|---|---|
#*   2|   |   |   |   |   |   |   |   |
#*    |---|---|---|---|---|---|---|---|
#*   3|   | ==|=> |   |   |   |   |   |
#*    |---|---|---|---|---|---|---|---|
#*   4|   |   |   |   |   |   |   |   |
#*    |---|---|---|---|---|---|---|---|
#*   5|   |   |   |   |   |   |   |   |
#*    |---|---|---|---|---|---|---|---|
#*   6|   |   |   |   |   |   |   |   |
#*    |---|---|---|---|---|---|---|---|
#*   7|   |   |   |   |   |   |   |   |
#*    |---|---|---|---|---|---|---|---|

#* SOURCE CELL / DEST CELL = (index of row, index of col)
#* IMAGE SHOULD BE OPEN CV 
def drawArrowFromSourceToDestCell(sourceCell, destCell, image):
    
    imageHeight, imageWidth, channels = image.shape

    def getCoordinateOfCenterOfCell(cell):
        i, j = cell
        left = j * imageWidth / 8
        right = (j + 1) * imageWidth / 8
        upper = i * imageHeight / 8
        lower = (i + 1) * imageHeight / 8

        center = (left + (right-left)/2, upper + (lower-upper)/2)
        return ( (int)(center[0]), (int)(center[1]) )

    sourceCenterCoordinate = getCoordinateOfCenterOfCell(sourceCell)
    destCenterCoordinate = getCoordinateOfCenterOfCell(destCell)

    # Draw arrowed line from <sourceCenterCoordinate> to <destCenterCoordinate> in color(rgb(0,0,0)) with thickness <arrowWidthPixels> pixels
    arrowWidthPixels = 5
    image = cv2.arrowedLine(image, sourceCenterCoordinate , destCenterCoordinate, (160,32,240), arrowWidthPixels)

    
    return image











#* returns (cell row, cell col) from moveString(algebric notation) (e.g. "e7e8q")
def getCellFromAlgebricMove(moveString) :
    fromSquare =  ( (8 - (ord(moveString[1]) - ord('0'))), (ord(moveString[0]) - ord('a')) ) 
    toSquare = ( (8 - (ord(moveString[3]) - ord('0'))), (ord(moveString[2]) - ord('a')) )

    return (fromSquare, toSquare)










# *returns squares[][] from given fen string 
def getSquaresFromFenString(fen) :

    squares = []
    for i in range(8) :
        currRow = []
        for j in range(8) :
            currRow.append(" ")

        squares.append(currRow)

    #//*update the positions of pieces according to fen string________________________________
    fenSize = len(fen)
    k = 0

    rank = 8
    file = 1
    while (k < fenSize and fen[k] != ' ') :
    
        if (fen[k].isdigit()) :
            file += int(fen[k]) 
            # make these squares empty
        
        elif (fen[k] == '/') :
            file = 1
            rank -= 1
        
        else :
        
            pieceVal = (fen[k])

            # # if invalid piece character inside fen
            # if (pieceVal == -1)
            #     throw runtime_error("invalid piece character of fen inside initializeFromFenString() function");

            #*set the piece
            # this->pieceBitBoards[pieceVal].setBitAt((8 - rank) * 8 + (file - 1));
            squares[8-rank][file-1] = pieceVal
            file += 1

            # if (++file > 9)
            #     throw runtime_error("file > 9 initializeFromFenString() function");
        

        # //*increment k
        k+=1
    
    return squares



    # if (rank != 1 || file != 9)
    #     throw runtime_error("rank!=1 || file!=9   inside initializeFromFenString() function");

    # #//*update other board state variables __________________________________________________________________
    # k+=1

    # if (k >= fenSize || !(fen[k] == 'w' || fen[k] == 'b'))
    #     throw runtime_error("invalid player color inside initializeFromFenString() function");

    # this->currentPlayer = fen[k] == 'w' ? WHITE : BLACK;

    # //*increment k
    # k++;

    # //*if next character is not white space
    # if (k >= fenSize || fen[k] != ' ')
    #     throw runtime_error("white space missing after currentPlayer in FEN inside initializeFromFenString() function");

    # //*ignore white space
    # k++;

    # //*castling availability
    # if (k >= fenSize)
    #     throw runtime_error("no castling availability inside initializeFromFenString() function");

    # if (fen[k] == '-')
    # {
    #     for (int i = 0; i < 4; i++)
    #         this->castlingRights[i] = false;

    #     // *skip the castling string
    #     k++;
    # }
    # else
    # {
    #     vector<char> foundChars = {};
    #     while (k < fenSize && fen[k] != ' ')
    #     {
    #         if (!(fen[k] == 'K' || fen[k] == 'Q' || fen[k] == 'k' || fen[k] == 'q'))
    #             throw runtime_error("invalid castling info inside initializeFromFenString() function");

    #         //*if one castling info is repeated
    #         for (char c : foundChars)
    #             if (c == fen[k])
    #                 throw runtime_error("one castling info repeated inside initializeFromFenString() function");

    #         foundChars.push_back(fen[k]);

    #         switch (fen[k])
    #         {
    #         case 'K':
    #             this->castlingRights[0] = true;
    #             break;
    #         case 'Q':
    #             this->castlingRights[1] = true;
    #             break;
    #         case 'k':
    #             this->castlingRights[2] = true;
    #             break;
    #         case 'q':
    #             this->castlingRights[3] = true;
    #             break;

    #         default:
    #             throw runtime_error("invalid castling info inside initializeFromFenString() function");
    #         }

    #         //*increment k
    #         k++;
    #     }

    #     if (k >= fenSize)
    #         throw runtime_error("problem on or after castling rights inside initializeFromFenString() function");
    # }

    # //*ignore the white space
    # if (k >= fenSize || fen[k] != ' ')
    #     throw runtime_error(" problem initializeFromFenString() function");
    # //*increment
    # k++;

    # //*check enPassant_________________________________________________________
    # if (k >= fenSize)
    #     throw runtime_error(" no enpassanti inside initializeFromFenString() function");

    # if (fen[k] == '-')
    # {
    #     this->enPassantSquareIndex = -1;
    #     k++;
    # }
    # else
    # {
    #     char enPassantFile = fen[k];
    #     k++;
    #     if (k >= fenSize)
    #         throw runtime_error(" no enpassanti rank inside initializeFromFenString() function");

    #     int enPassantRank = fen[k] - '0';
    #     this->enPassantSquareIndex = (8 - enPassantRank) * 8 + (enPassantFile - 'a');
    #     if (this->enPassantSquareIndex < 0 || this->enPassantSquareIndex >= 64)
    #         throw runtime_error(" invalid enpassanti coordinate inside initializeFromFenString() function");

    #     k++;
    # }

    # //*check enPassant square validity
    # if (this->enPassantSquareIndex != -1)
    # {
    #     if (this->currentPlayer == BLACK)
    #     {
    #         if (BitBoard::getRankOfSquareIndex(this->enPassantSquareIndex) != 3)
    #             throw runtime_error("\n\ninvalid enPassant square given in fen string -- from Board.initializeFromFenString()\n\n");

    #         //*if there is no pawn in front of enPassant square
    #         if (this->pieceBitBoards[Piece::P].getBitAt(this->enPassantSquareIndex - 8) == 0)
    #             throw runtime_error("\n\ninvalid enPassant square given in fen string -- from Board.initializeFromFenString()\n\n");
    #     }
    #     else if (this->currentPlayer == WHITE)
    #     {
    #         if (BitBoard::getRankOfSquareIndex(this->enPassantSquareIndex) != 6)
    #             throw runtime_error("\n\ninvalid enPassant square given in fen string -- from Board.initializeFromFenString()\n\n");

    #         //*if there is no pawn in front of enPassant square
    #         if (this->pieceBitBoards[Piece::p].getBitAt(this->enPassantSquareIndex + 8) == 0)
    #             throw runtime_error("\n\ninvalid enPassant square given in fen string -- from Board.initializeFromFenString()\n\n");
    #     }
    # }

    # //*TODO: HAlf move and full move remaining





