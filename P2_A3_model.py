import numpy as np
import matplotlib.pyplot as plt
import time as t
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras import models
from tensorflow.keras.layers import LSTM, Dense, Flatten#, Conv2d, MaxPooling2D

#from tensorflow.keras.optimizers import SGD
#from tensorflow.keras.callbacks import ReduceLROnPlateau

tstart = t.time()

split = 0.3
seed = 42
adam = Adam(learning_rate=0.001)

#LOAD DATA SETS
X = np.load("./P2_A_training_data.npy",allow_pickle=True)
y = np.load("./P2_A_labels.npy")

#USE ONLY MOTION DATA
X = X[:,:,:6]

#SPLIT INTO TEST AND TRAINING DATASETS
X_train, X_test, y_train, y_test = train_test_split(X,  y, test_size = split, random_state = seed)

print(X_train.shape)
#print(X_test.shape)
print(y_train.shape)
#print(y_test.shape)

print(X_train[1,:,:])
print(y_train[1,:])


#X_train = np.reshape(a=X_train, newshape=(X_train.shape[0], X_train.shape[1]*X_train.shape[2]))
#X_test = np.reshape(a=X_test, newshape=(X_test.shape[0], X_test.shape[1]*X_test.shape[2]))

#print(X_train.shape)
#print(X_test.shape)


#BUILD MODEL

val_split = 0.3
epochs = 50
loss = CategoricalCrossentropy(from_logits=True)

#BUILD MODEL
def get_model():
    model = Sequential()
    model.add(Flatten())
    #model.add(GaussianNoise(0.1, name = 'Gaussian'))
    #model.add(Input(shape=X_train.shape[1:]))
    #model.add(LSTM(units=8, activation='relu', return_sequences=True, name='LSTM1'))
    #model.add(LSTM(units=8, activation='relu', name = 'LSTM2'))
    #model.add(LSTM(units=16, activation='tanh', return_sequences=True, name = 'LSTM1'))
    #model.add(LSTM(units=16, activation='tanh', return_sequences=True, name = 'LSTM2'))
    #model.add(LSTM(units=8, activation='tanh', name = 'LSTM3'))
    #model.add(Dense(units=32, activation='relu', name = 'Dense1'))
    model.add(Dense(units=32, activation='relu', name = 'Dense2'))
    model.add(Dense(units=16, activation='relu', name = 'Dense3'))
    #model.add(Dense(units=16, activation='sigmoid', name = 'Dense4'))
    model.add(Dense(units=8, activation='relu', name = 'Dense5'))
    #model.add(Dense(units=8, activation='sigmoid', name = 'Dense6'))
    model.add(Dense(units=4, activation='relu', name = 'Dense7'))
    model.add(Dense(units=2, activation='softmax', name = 'Dense8'))

    model.compile(loss = loss, optimizer=adam, metrics = ['accuracy'])
    return model

model = get_model()
print("MODEL BUILT")
X_train=np.asarray(X_train).astype(np.float32)
y_train=np.asarray(y_train).astype(np.int32)
hist = model.fit(X_train, y_train, validation_split=val_split, epochs=epochs)
print("MODEL TRAINED")
print(model.summary())

#SAVE TRAINING PERFORMANCE
np.save("./P2_A_ML_Project_performance.npy", hist.history)
print("MODEL TRAINING PERFORMANCE SAVED")


"""
model.save("./model_weights.h5")
"""

#TEST MODEL PERFORMANCE ON TEST DATA
X_test=np.asarray(X_test).astype(np.float32)
y_test=np.asarray(y_test).astype(np.int32)

model.evaluate(X_test, y_test)
y_pred = model.predict(X_test)

#y_true=y_test
y_pred = np.argmax(y_pred, axis=1)
y_pred = np.reshape(y_pred, (len(y_pred),1))
y_true = np.argmax(y_test, axis=1)
print(y_pred.shape)
print(y_true.shape)
np.save("./P2_A_y_true.npy", arr=y_true)
np.save("./P2_A_y_pred.npy", arr=y_pred)
print("PREDICTED AND TRUE LABELS SAVED")

tend = t.time()
print("RUNTIME: %.2f s"%(tend-tstart))
"""
cm = confusion_matrix(y_label, y_pred)#, normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[-2,-1,0,1])
disp.plot()
plt.savefig("./confusion_matrix_test.png")
print("confusion matrix test")


#train
y_pred = model.predict(X_train)
y_pred = np.argmax(y_pred, axis=1)
y_label = np.argmax(y_train, axis=1)
print('pred')
cm = confusion_matrix(y_label, y_pred)#, normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[-2,-1,0,1])
disp.plot()
print('confusion matrix train')
plt.savefig("./confusion_matrix_train.png")
"""

"""
prediction = model.predict(X_test)
model.evaluate(X_test, y_test)

hist = np.load("./ML_Project_performance.npy", allow_pickle=True)

tend = t.time()
"""
print("Model Runtime: %.2f s"%(tend-tstart))

"""
plt.plot(np.arange(epochs)+1, hist[()]['loss'],label='mse loss (train)')
plt.plot(np.arange(epochs)+1, hist[()]['val_loss'], label='mse loss (val)')
plt.title('MSE loss performance over %d epochs'%(epochs))
plt.xlabel('# of epochs')
plt.legend()
plt.savefig("./performance_loss.png")
"""

"""
plt.plot(np.arange(epochs)+1, hist[()]['accuracy'],label='accuracy (train)')
plt.plot(np.arange(epochs)+1, hist[()]['val_accuracy'], label='accuracy (val)')
plt.title('Accuracy performance over %d epochs'%(epochs))
plt.xlabel('# of epochs')
plt.legend()
plt.savefig("./performance_accuracy.png")
"""