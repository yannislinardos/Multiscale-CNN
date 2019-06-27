from utils import *
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dropout, Conv1D, Flatten, Dense, AveragePooling1D, MaxPooling1D
import gc


khz = 24

input_dim = 4 * khz * 1000

model = Sequential()
model.add(Conv1D(16, kernel_size=9, activation='relu', input_shape=(input_dim, 1), padding='same'))
model.add(AveragePooling1D(pool_size=2))

model.add(Conv1D(16, kernel_size=9, activation='relu', input_shape=(input_dim, 1), padding='same'))
model.add(AveragePooling1D(pool_size=2))

model.add(Conv1D(16, kernel_size=9, activation='relu', input_shape=(input_dim, 1), padding='same'))
model.add(AveragePooling1D(pool_size=2))

# model.add(Conv1D(16, kernel_size=11, activation='relu', padding='same'))
# model.add(MaxPooling1D(pool_size=2))

# model.add(Conv1D(8, kernel_size=9, activation='relu', padding='same'))
# model.add(MaxPooling1D(pool_size=2))

# model.add(MaxPooling1D(pool_size=2))

model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(5, activation='sigmoid'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

X, Y = load_data('Dataframes/Training{}.pickle'.format(khz))

overfitCallback = EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', baseline=None,
                                restore_best_weights=True)

model.fit(X, Y, batch_size=1, epochs=3, shuffle=True, validation_split=0.12, callbacks=[overfitCallback])

gc.collect()

X, Y = load_data('Dataframes/Testing{}.pickle'.format(khz))

score = model.evaluate(X, Y, verbose=1, batch_size=3)
print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))

#save_model(model, 'Model_48KHz')