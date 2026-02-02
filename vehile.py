import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import os

base_path = "Vehicles"

for dirname, _, filenames in os.walk(base_path):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import os 
import cv2 
from PIL import Image
from tensorflow import keras 
from keras.models import Sequential 
from keras.layers import Dense , Conv2D , MaxPooling2D , Flatten , Dropout
from keras.utils import to_categorical
from keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
import warnings 
warnings.filterwarnings("ignore")

X = []
y = []
img_size = 224 
labels = ["Auto Rickshaws" , "Bikes" , "Cars" , "Motorcycles" , "Planes" , "Ships" , "Trains"]

for label in labels : 
    folder_path = os.path.join("Vehicles", label)
    for path in os.listdir(folder_path) : 
        img_path = os.path.join(folder_path , path)
        img = Image.open(img_path)
        img = img.convert("RGB")
        img = img.resize((img_size, img_size))
        X.append(img)
        y.append(label)

X = np.array(X)
y = np.array(y)

print("X : " , X.shape)
print("y : " , y.shape)

X_train , X_test , y_train , y_test = train_test_split(X ,y , test_size = 0.2 , random_state = 42)

X_train , y_train = shuffle(X_train , y_train , random_state = 42) 

X_train.shape

y_train.shape

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

datagen = ImageDataGenerator(
    rescale = 1 / 255 ,
    rotation_range = 90 ,
    zoom_range = 0.2 , 
    width_shift_range = 0.2 ,
    height_shift_range = 0.2 , 
    horizontal_flip = True , 
)

datagen.fit(X_train)


model  = Sequential()

model.add(Conv2D(filters = 8 , kernel_size = (3,3) , padding = "same" , 
                activation = "relu" , input_shape = (224,224,3)))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(filters = 16, kernel_size = (3,3) , padding = "same" , 
                activation = "relu" ))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())
model.add(Dense(128 , activation = "relu"))
model.add(Dense(128 , activation = "relu"))
model.add(Dense(64 , activation = "relu"))
model.add(Dense(7 , activation = "softmax"))

learning_rate = 0.001
optimizer = Adam(learning_rate = learning_rate)

model.compile(optimizer = optimizer , loss = "categorical_crossentropy" , metrics = ["accuracy"])

history = model.fit(X_train , y_train , epochs = 20 , batch_size = 32 , validation_data = (X_test , y_test) )

model.summary()

from sklearn.metrics import classification_report

y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred , axis = 1)
y_test = np.argmax(y_test , axis = 1)

print(classification_report(y_pred,y_test))



