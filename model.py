# IMPORTING LIBRARIES
import keras
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import BatchNormalization
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense

# FETCHING TRAINING AND TESTING IMAGES AND PREPROCESSING IMAGES
image_gen=ImageDataGenerator(rescale=1./255)                         
 
train_image_gen=image_gen.flow_from_directory("data/seg_train/seg_train/",target_size=(150,150),batch_size=16,class_mode="categorical")    

test_image_gen=image_gen.flow_from_directory("data/seg_train/seg_train/",target_size=(150,150),batch_size=16,class_mode="categorical")    

# GETTING INDICES OF CATEGORIES
print(train_image_gen.class_indices)

# BUILDING MODEL
model = Sequential()
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(6,activation='softmax'))

# COMPILATION USING CATEGORICAL CROSSENTROPY
model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])

# TRAINING THE MODEL
results=model.fit_generator(train_image_gen,epochs=20,steps_per_epoch=150,validation_data=test_image_gen)

# SAVING MODEL
model.save("model.h5")

# VISUALIZATION OF ACCURACIES
plt.plot(results.history['accuracy'],label="accuracy")
plt.plot(results.history['val_accuracy'],label="val_accuracy")
plt.legend()
plt.show()