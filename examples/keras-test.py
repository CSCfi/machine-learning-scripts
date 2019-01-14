
import keras
print('Keras version:', keras.__version__, '(version 2 or more required)')

from keras.models import Sequential

model = Sequential()

from keras.layers import Dense, Activation

model.add(Dense(units=64, input_dim=100))
model.add(Activation("relu"))
model.add(Dense(units=10))
model.add(Activation("softmax"))

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

print(model.summary())
