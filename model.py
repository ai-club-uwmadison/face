from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

class Model :

	def __init__ (self) :
		self.model = self.createModel ()

	def createModel (self) :
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
		model.add (Dense (6))

		model.compile (loss='categorical_crossentropy',
						optimizer='adam',
						metrics=['accuracy'])

		print (model.summary ())

		return model

	def train (self, X_train, y_train) :
		model.fit (X_train, y_train, epochs=50, batch_size=32)

model = Model ()