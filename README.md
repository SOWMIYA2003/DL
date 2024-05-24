# DL

https://github.com/SOWMIYA2003/basic-nn-model

## Developing a Neural Network Regression Model

```
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
