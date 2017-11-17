import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

class Model :

	def __init__ (self) :
		pass

	def init (self) :
		model = Sequential()
		model.add (Conv2D (32, (3, 3), input_shape=(128, 128, 3), padding='same'))
		model.add (Activation ('relu'))
		model.add (MaxPooling2D (pool_size=(2, 2)))
		model.add (Dropout (0.5))

		model.add (Conv2D (64, (3, 3), padding='same'))
		model.add (Activation ('relu'))
		model.add (MaxPooling2D (pool_size=(2, 2)))
		model.add (Dropout (0.5))

		model.add (Conv2D (128, (3, 3), padding='same'))
		model.add (Activation ('relu'))
		model.add (MaxPooling2D (pool_size=(2, 2)))
		model.add (Dropout (0.5))

		model.add (Conv2D (256, (3, 3), padding='same'))
		model.add (Activation ('relu'))
		model.add (MaxPooling2D (pool_size=(2, 2)))
		model.add (Dropout (0.5))

		model.add (Conv2D (512, (3, 3), padding='same'))
		model.add (Activation ('relu'))
		model.add (MaxPooling2D (pool_size=(2, 2)))
		model.add (Dropout (0.5))

		model.add (Flatten ())
		model.add (Dense (1024))
		model.add (Activation ('relu'))
		model.add (Dropout (0.5))
		model.add (Dense (256))
		model.add (Activation ('relu'))
		model.add (Dropout (0.5))
		model.add (Dense (64))
		model.add (Activation ('relu'))
		model.add (Dropout (0.5))
		model.add (Dense (16))
		model.add (Activation ('relu'))
		model.add (Dropout (0.5))
		model.add (Dense (4))

		model.compile (loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

		print (model.summary ())

		self.model = model

	def train (self, X_train, y_train, X_val, y_val, epochs=5, batch_size=32) :
		self.model.fit (X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))

	def evaluate (self, X, y) :
		loss, acc = self.model.evaluate (X, y)
		print ('Model evaluation >> loss: {:.3f} - acc: {:.3f}'.format (loss, acc))

	def predict (self, X, classes=[]) :
		X_new = X.reshape (1, X.shape[0], X.shape[1], X.shape[2])
		label = np.argmax (self.model.predict (X_new))

		if len (classes) == 0 :
			return label
		else :
			return classes[label]