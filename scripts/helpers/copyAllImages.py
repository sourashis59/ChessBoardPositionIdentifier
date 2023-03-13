# ======================================OBJECTIVE====================================
# copy all images from "data/rawData/Lichess" into "data/trainingData/allImages"
# ===================================================================================





import os
from PIL import Image




sourcePath = "data/rawData/Lichess"
destPath = "data/trainingData/allImages"


print("Copying images from \'" + sourcePath + "\', to \'" + destPath + "\' ...........")

# if we want to copy images from specific directories, instead of all directories
copyAllImages = False

# right now, i am only includeing the obvious looking piece images into the data set to make the learning less complex
# later i have to include the other rare types of pieces into the data set 
directoriesToInclude = ["piecetype1", "piecetype2","piecetype3","piecetype17","piecetype18","piecetype19" ]

sourceDirectory = os.fsencode(sourcePath)
for subDir in os.listdir(sourceDirectory):

    subDirName = os.fsdecode(subDir)
    subDirPath = os.fsdecode(f"{sourcePath}/{subDirName}")

    if(copyAllImages == False and subDirName not in directoriesToInclude)  :
        continue

    # there are "piecetype{i}" subdirectories inside Lichess, iterate throught each subdir, and find the files
    if( os.path.isdir(subDirPath) ) :
        for file in os.listdir(subDirPath) :
            fileName = os.fsdecode(file)
            if ( fileName.endswith(".jpg") or fileName.endswith(".png") ) :                 
                # print(os.path.join(subDirPath, fileName))
                currImage = Image.open( os.path.join(subDirPath, fileName) )
                currImage.save(f"{destPath}/{fileName}")




print("\nCopying Completed !\n")