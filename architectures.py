import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Conv2D, BatchNormalization, Activation, Dense, Flatten, Dropout, add
from keras.regularizers import l2
from keras import Input, callbacks
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import pickle

def build_net(filters=256):
	board_inputs = Input(shape=(8, 8, 2))
	x = board_inputs
	x = Conv2D(filters, 3, padding='same', use_bias=False)(x)
	x = BatchNormalization(axis=3)(x)
	conv_block_1 = Activation('relu')(x)

	x = Conv2D(filters, 3, padding='same', use_bias=False)(conv_block_1)
	x = BatchNormalization(axis=3)(x)
	x = Activation('relu')(x)
	x = Conv2D(filters, 3, padding='same', use_bias=False)(x)
	x = BatchNormalization(axis=3)(x)
	x = add([x, conv_block_1])
	res_block_1 = Activation('relu')(x)

	x = Conv2D(filters, 3, padding='same', use_bias=False)(res_block_1)
	x = BatchNormalization(axis=3)(x)
	x = Activation('relu')(x)
	x = Conv2D(filters, 3, padding='same', use_bias=False)(x)
	x = BatchNormalization(axis=3)(x)
	x = add([x, res_block_1])
	res_block_2 = Activation('relu')(x)

	x = Conv2D(filters, 3, padding='same', use_bias=False)(res_block_2)
	x = BatchNormalization(axis=3)(x)
	x = Activation('relu')(x)
	x = Conv2D(filters, 3, padding='same', use_bias=False)(x)
	x = BatchNormalization(axis=3)(x)
	x = add([x, res_block_2])
	res_block_3 = Activation('relu')(x)

	x = Conv2D(filters, 3, padding='same', use_bias=False)(res_block_3)
	x = BatchNormalization(axis=3)(x)
	x = Activation('relu')(x)
	x = Conv2D(filters, 3, padding='same', use_bias=False)(x)
	x = BatchNormalization(axis=3)(x)
	x = add([x, res_block_3])
	res_block_4 = Activation('relu')(x)

	policy = Conv2D(2, 1, use_bias=False)(res_block_4)
	policy = BatchNormalization(axis=3)(policy)
	policy = Activation('relu')(policy)
	policy = Flatten()(policy)
	# policy = Dense(512, use_bias=False)(policy)
	# policy = BatchNormalization(axis=1)(policy)
	# policy = Activation('relu')(policy)
	# policy = Dropout(0.3)(policy)
	# policy = Dense(256, use_bias=False)(policy)
	# policy = BatchNormalization(axis=1)(policy)
	# policy = Activation('relu')(policy)
	policy = Dropout(0.3)(policy)
	policy = Dense(65, activation='softmax')(policy)

	value = Conv2D(1, 1, use_bias=False)(res_block_4)
	value = BatchNormalization(axis=3)(value)
	value = Activation('relu')(value)
	value = Flatten()(value)
	# value = Dense(1024, use_bias=False)(value)
	# value = BatchNormalization(axis=1)(value)
	# value = Activation('relu')(value)
	# value = Dropout(0.3)(value)
	value = Dense(256, use_bias=False)(value)
	value = BatchNormalization(axis=1)(value)
	value = Activation('relu')(value)
	value = Dropout(0.3)(value)
	value = Dense(1, activation='tanh')(value)

	model = Model(inputs=board_inputs, outputs=[value, policy], name='Othello')
	return model

def compile_net(filters=256):
	net = build_net(filters)
	net.summary()

	net.compile(loss=['mean_squared_error', 'categorical_crossentropy'], optimizer=Adam(0.001), metrics=['accuracy'])
	net.save('networks/othello_test.h5')
	return net

def prep_data():
	with open('train_data.txt', 'rb') as fp:
		data = pickle.load(fp)
	boards, policy, outcome = zip(*data)
	X = np.concatenate(boards)
	value = np.asarray(outcome)
	pol = np.asarray(policy)
	X_train, X_test, value_train, value_test, pol_train, pol_test = train_test_split(X, value, pol, test_size=0.2, random_state=42)
	X_val, X_test, value_val, value_test, pol_val, pol_test = train_test_split(X_test, value_test, pol_test, test_size=0.5, random_state=42)
	return X_train, X_val, X_test, value_train, value_val, value_test, pol_train, pol_val, pol_test

my_callbacks = [
	callbacks.EarlyStopping(patience=5),
	callbacks.ModelCheckpoint(filepath='networks/best_params.h5', verbose=1, save_best_only=True)
]

#for testing network architectures
def train_net(X_train, X_val, value_train, value_val, pol_train, pol_val, batch_size=16):
	net = keras.models.load_model('networks/othello_best.h5')
	history = net.fit(X_train, [value_train, pol_train], epochs=50, batch_size=batch_size,
		validation_data=(X_val, [value_val, pol_val]), callbacks=my_callbacks)

def train():
	X_train, X_val, X_test, value_train, value_val, value_test, pol_train, pol_val, pol_test = prep_data()
	train_net(X_train, X_val, value_train, value_val, pol_train, pol_val)


