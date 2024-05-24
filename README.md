# DL



## Developing a Neural Network Regression Model

https://github.com/SOWMIYA2003/basic-nn-model

```
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

Model = Sequential ([
    Dense(units = 5, activation ='relu', input_shape = [1]),
    Dense(units = 3, activation ='relu'),
    Dense(units = 4, activation ='relu'),
    Dense(units=1)
])

Model.compile(optimizer='rmsprop',loss='mse')

Model.fit(x=X_train1,y=y_train,epochs=2000)
```

## Developing a Neural Network Classification Model

https://github.com/SOWMIYA2003/nn-classification

```
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np
import matplotlib.pylab as plt
```
```
Model = Sequential ([
    Dense(units = 3, activation ='relu', input_shape = [8]),
    Dense(units = 2, activation ='relu'),
    Dense(units=4,activation = 'softmax')
])

Model.compile(optimizer='adam',
                 loss= 'categorical_crossentropy',
                 metrics=['accuracy'])

Model.fit(x=X_train_scaled,y=y_train,
             epochs= 2000,
             batch_size=256,
             validation_data=(X_test_scaled,y_test),
             )
```
## Convolutional Deep Neural Network for Digit Classification

https://github.com/SOWMIYA2003/mnist-classification
```
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
```
```
from tensorflow.keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

model = keras.Sequential([
    layers.Input(shape=(28,28,1)),
    layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu'),
    layers.MaxPool2D(pool_size=(2,2)),
    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics='accuracy')
model.fit(X_train_scaled ,y_train_onehot, epochs=5,
          batch_size=64,
          validation_data=(X_test_scaled,y_test_onehot))
```
## Deep Neural Network for Malaria Infected Cell Recognition

https://github.com/SOWMIYA2003/malaria-cell-recognition

```
from tensorflow import keras
from tensorflow.keras import layers
model = keras.Sequential([
    layers.Input(shape=image_shape),
    layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu'),
    layers.MaxPool2D(pool_size=(2,2)),
    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
model.summary()
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

batch_size = 16
train_image_gen = image_gen.flow_from_directory(train_path,
                                               target_size=image_shape[:2],
                                                color_mode='rgb',
                                               batch_size=batch_size,
                                               class_mode='binary')
test_image_gen = image_gen.flow_from_directory(test_path,
                                               target_size=image_shape[:2],
                                               color_mode='rgb',
                                               batch_size=batch_size,
                                               class_mode='binary',shuffle=False)
                                        train_image_gen.class_indices
results = model.fit(train_image_gen,epochs=20,
                              validation_data=test_image_gen)
```


## Stock Price Prediction

https://github.com/SOWMIYA2003/rnn-stock-price-prediction

```
model = Sequential([
    layers.SimpleRNN(50, input_shape=(60, 1)),
    layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.summary()
model.fit(X_train1,y_train,epochs=100, batch_size=32)
```

## Named Entity Recognition

https://github.com/SOWMIYA2003/named-entity-recognition

```
max_len = 50
num_words = len(words)
input_word = layers.Input(shape=(max_len,))
embedding_layer = layers.Embedding(input_dim = num_words,
                                   output_dim = 50,
                                   input_length = max_len)(input_word)
dropout_layer = layers.SpatialDropout1D(0.1)(embedding_layer)
bidirectional_lstm = layers.Bidirectional(layers.LSTM(
    units=100, return_sequences=True,recurrent_dropout=0.1))(dropout_layer)
output = layers.TimeDistributed(
    layers.Dense(num_tags, activation="softmax"))(bidirectional_lstm)
model = Model(input_word, output)    

model.summary()
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

history = model.fit(
    x=X_train,
    y=y_train,
    validation_data=(X_test,y_test),
    batch_size=32,
    epochs=3,
)
```

## Convolutional Autoencoder for Image Denoising

https://github.com/SOWMIYA2003/convolutional-denoising-autoencoder

```
input_img = keras.Input(shape=(28, 28, 1))
# Write your encoder here
x = layers.Conv2D(16, (3,3), activation = 'relu', padding='same') (input_img)
x =layers.MaxPooling2D((2,2), padding='same') (x)
x = layers.Conv2D(8, (3,3), activation = 'relu', padding='same') (x)
x = layers. MaxPooling2D((2,2), padding='same') (x)
x =layers.Conv2D(8, (3,3), activation = 'relu', padding='same')(x)

encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

# Encoder output dimension is ## Mention the dimention ##

x = layers.Conv2D(8, (3,3), activation = 'relu',padding='same') (encoded)
x = layers. UpSampling2D((2,2))(x)
x = layers.Conv2D(8, (3,3),activation = 'relu', padding='same') (x)
x = layers. UpSampling2D((2,2))(x)
x = layers.Conv2D(16, (3,3), activation = 'relu')(x)
x = layers. UpSampling2D((2,2))(x)

# Write your decoder here

decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = keras.Model(input_img, decoded)
autoencoder.summary()
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(x_train_noisy, x_train_scaled,
                epochs=2,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test_noisy, x_test_scaled))
```
