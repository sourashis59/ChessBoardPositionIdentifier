import time
import tensorflow as tf
import os 
import cv2
# import imghdr
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.metrics import Precision, Recall , BinaryAccuracy


gpus = tf.config.experimental.list_physical_devices("GPU")
print(f"gpus = {gpus}\n\n")




collabSourcePath = ""
resizedImageDimension = (64, 64)

# ======================================================================================================================
# ====================================================== LOAD DATA =====================================================
# ======================================================================================================================

# dataPath = os.path.join(collabSourcePath, "models", "emptyNonemptySquareClassifier", "data", "trainingData", "images")
dataPath = os.path.join("data/trainingData/emptyNonemptySquares")

trainedModelDestPath = os.path.join(collabSourcePath, "models", "emptyNonemptySquareClassifier", "trainedModelCNN.h5")

emptySquareSourcePath = 'data/trainingData/images/emptySquares'    
nonemptySquareSourcePath = 'data/trainingData/images/nonemptySquares'    


# for squareTypeInd, squareType in enumerate(squareTypes) :
    
#     filePath = ""
#     if(squareType == 'empty'):
#         filePath = emptySquareSourcePath
#     else:
#         filePath = nonemptySquareSourcePath 

#     for file in os.listdir(filePath):
#         imagePath = os.path.join(filePath, file)
#         # image = Image.open(imagePath)
#         # image = image.resize((20,20))
#         # data.append( list(image.getdata()) )
        
#         # image = imread(imagePath)
#         image = cv2.imread(imagePath)
        
#         # # make image black and white
#         # # Only convert image to grayscale if it is RGB
#         # if image.shape[-1] == 3:
#         #     image = rgb2gray(image)

#         image = resize(image, resizedImageDimension)
#         data.append(image.flatten())
        
        
#         labels.append(squareTypeInd)





print("\n\nPreparing data...............\n")
# *this also shuffles the data
# * data contains batches of {image, label}
# * image is resized to <resizedImageDimension> and converted to "RGB"
data = tf.keras.utils.image_dataset_from_directory(dataPath, image_size=resizedImageDimension)


print("Data prepared...............\n")


# print(f"\n\nclass names : {data.class_names}" )  
classes = data.class_names




## batch contains array of data and labels. (data means rgb image)
# batch = data.as_numpy_iterator().next()
# # print(batch[1]) #it is labels[]
# print(batch[0][0].shape)
# print(batch[0][0])

# cv2.imwrite("models/emptyNonemptySquareClassifier/test/ultra_test_cv2.png", batch[0][0])

# # tempImage = batch[0][0]
# tempImage = cv2.imread("models/emptyNonemptySquareClassifier/test/ultra_test_cv2.png")
# print(tempImage)
# plt.imshow(tempImage)    
# plt.show()


#* scale the image pixel values from range [0, 255] to [0,1] ==> this range helps the Depp learning model learn faster
# batch[0] = batch[0] / 255

# x is image, y is label
data = data.map(lambda x,y : (x/255, y)) 
# print(data.as_numpy_iterator().next()[0].max())
# print(data.as_numpy_iterator().next()[0])



# batch = data.as_numpy_iterator().next()
# tempImage = batch[0][0] * 255
# cv2.imwrite("models/emptyNonemptySquareClassifier/test/ultra_test_cv2.png", tempImage)
# # tempImage = batch[0][0]
# tempImage = cv2.imread("models/emptyNonemptySquareClassifier/test/ultra_test_cv2.png")
# print(tempImage)
# plt.imshow(tempImage)    
# plt.show()








# ======================================================================================================================
# ================================================= TRAIN TEST SPLIT   =================================================
# ======================================================================================================================

# print(len(data))
trainSize = int(len(data) * 0.7) #70%
validationSize = int(len(data) * 0.2 ) 
testSize = int(len(data) * 0.1) + 1

# print(f"(no of batches) : trainingSize = {trainSize}, validationSize = {validationSize}, testSize = {testSize}")
trainData = data.take(trainSize)
validationData = data.skip(trainSize).take(validationSize)
testData = data.skip(trainSize + validationSize).take(testSize)
# print(f"(no of batches) : trainingSize = {len(trainData)}, validationSize = {len(validationData)}, testSize = {len(testData)}")
















# ======================================================================================================================
# ================================================= TRAIN THE MODEL   =================================================
# ======================================================================================================================
model = Sequential()

# 16 features, filter matrix dimension = 3*3
model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape = ( resizedImageDimension[0], resizedImageDimension[1], 3)))
model.add(MaxPooling2D())

model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(256, activation = 'relu'))
model.add(Dense(len(classes), activation='sigmoid'))


model.compile('adam', loss="sparse_categorical_crossentropy", metrics=['accuracy'])

print("\n\nmodel  : \n" )
print(model.summary())
print("\n\n")


# # log while training the model
# logDirectory = "logs"
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logDirectory)

startTime = time.process_time()
print("\n\nStarting training.............................")

history = model.fit(trainData, epochs=20, validation_data = validationData )

endTime = time.process_time()
print("Elapsed time : ", (endTime - startTime) , " seconds.")
print("\n\n") # for new line








#* plot performance
fig = plt.figure()
plt.plot(history.history['loss'], color = 'red', label = 'loss')
plt.plot(history.history['val_loss'], color='orange', label='val_loss' )
fig.suptitle('Loss', fontsize = 20)
plt.legend(loc='upper left')
plt.show()










# ======================================================================================================================
# ================================================= TEST PERFORMANCE   =================================================
# ======================================================================================================================

print("\n\n\ntesting Performance..................\n")
precision = Precision()
recall = Recall()
accuracy = BinaryAccuracy()

for batch in testData.as_numpy_iterator() :
    
    # * y will be an array of labels (for example: y = [1,1,0,0,1,0,1,1,1,1,0,0])
    # * model.predict(X) will return 32*2 array of {probability of class1, prob of class2} (batch size = 32)
    # *     (for example : model.predict(x) = [ [0.2, 0.8], [0.5,0.5], [1, 0], [0.3, 0.7] ])
    # * np.argmax(arr) returns the index of the max element
    X, y = batch
    y_pred = [ np.argmax(element) for element in model.predict(X)] #* for each probablity (size 2 array), find the index of max 

    precision.update_state(y, y_pred)
    recall.update_state(y, y_pred)
    accuracy.update_state(y, y_pred)


print(f"\nprecision = {precision.result().numpy()}, Recall = {recall.result().numpy()}, Accuracy = {accuracy.result().numpy()}")









# ======================================================================================================================
# ================================================= SAVE THE MODEL   =================================================
# ======================================================================================================================
model.save( trainedModelDestPath  )

# model = load_model(trainedModelDestPath)






