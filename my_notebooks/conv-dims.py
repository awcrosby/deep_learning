from keras.models import Sequential
from keras.layers import Conv2D

model = Sequential()
model.add(Conv2D(filters=2, kernel_size=2, strides=1, padding='valid',
    activation='relu', input_shape=(200, 200, 1)))
model.summary()
