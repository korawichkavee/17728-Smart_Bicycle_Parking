import numpy as np
import os
import gc
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
import time as t

tstart = t.time()

#LOAD DATASETS
data_11_14_22 = np.load("P2_A_data_11-14-22.npy")
data_11_15_22 = np.load("P2_A_data_11-15-22.npy")
data_11_16_22 = np.load("P2_A_data_11-16-22.npy")
print("DATA LOADED")

print(data_11_14_22.shape)
print(data_11_15_22.shape)
print(data_11_16_22.shape)

step_rate = 1
duration_aim = 5 #sec
sample_rate = 8000
window_size = duration_aim*sample_rate # 20000/4000 = 5 --> 5/2 = 2.5 sec
mu = 0

#LABELING FUNCTION - 0: "-1", 1: "+1"
def labeling(arr):
  if ~arr.any():
    return 99
  else:
    arr = arr[arr!=0]
    vals, counts = np.unique(arr, return_counts=True)
    l = np.argmax(counts)
    v = vals[l]>0
    return 1*v

def noise():
  return 0

### SAMPLE DATA INTO BINS OF (WINDOW_SIZE X # OF FEATURES) ###

#BIN 11-14-22 DATA
bins_11_14_22 = []
labels_11_14_22 = []
sigma = np.std(data_11_14_22[:,1:-1], axis=0)
print(sigma)
for i in np.arange(0, len(data_11_14_22)//window_size * window_size - window_size * (1-1/step_rate), window_size/step_rate):
  i = int(i)
  labels = data_11_14_22[i:i+window_size,1]
  label = labeling(labels)
  if label != 99:
    feats = data_11_14_22[i:i+window_size,1:-1]
    noisy_feats_1 = feats
    noisy_feats_2 = feats
    for f in np.arange(feats.shape[1]):
      noisy_feats_1[:,f] = feats[:,f] + np.random.normal(mu, sigma[f], feats[:,f].shape)
      noisy_feats_2[:,f] = feats[:,f] + np.random.normal(mu, sigma[f], feats[:,f].shape)

    bins_11_14_22.append(feats)
    bins_11_14_22.append(noisy_feats_1)
    bins_11_14_22.append(noisy_feats_2)

    labels_11_14_22.append(label)
    labels_11_14_22.append(label)
    labels_11_14_22.append(label)
  else:
    continue
print("BINNED 11-14-22")


#BIN 11-15-22 DATA
bins_11_15_22 = []
labels_11_15_22 = []
sigma = np.std(data_11_15_22[:,1:-1], axis=0)
print(sigma)
for i in np.arange(0, len(data_11_15_22)//window_size * window_size - window_size * (1-1/step_rate), window_size/step_rate):
  i = int(i)
  labels = data_11_15_22[i:i+window_size,1]
  label = labeling(labels)
  if label != 99:
    feats = data_11_15_22[i:i+window_size,1:-1]
    noisy_feats_1 = feats
    noisy_feats_2 = feats

    for f in np.arange(feats.shape[1]):
      noisy_feats_1[:,f] = feats[:,f] + np.random.normal(mu, sigma[f], feats[:,f].shape)
      noisy_feats_2[:,f] = feats[:,f] + np.random.normal(mu, sigma[f], feats[:,f].shape)

    bins_11_15_22.append(feats)
    bins_11_15_22.append(noisy_feats_1)
    bins_11_15_22.append(noisy_feats_2)

    labels_11_15_22.append(label)
    labels_11_15_22.append(label)
    labels_11_15_22.append(label)
  else:
    continue
print("BINNED 11-15-22")

#BIN 11-16-22 DATA
bins_11_16_22 = []
labels_11_16_22 = []
sigma = np.std(data_11_16_22[:,1:-1], axis=0)
print(sigma)
for i in np.arange(0, len(data_11_16_22)//window_size * window_size - window_size * (1-1/step_rate), window_size/step_rate):
  i = int(i)
  labels = data_11_16_22[i:i+window_size,1]
  label = labeling(labels)
  if label != 99:
    feats = data_11_16_22[i:i+window_size,1:-1]
    noisy_feats_1 = feats
    noisy_feats_2 = feats

    for f in np.arange(feats.shape[1]):
      noisy_feats_1[:,f] = feats[:,f] + np.random.normal(mu, sigma[f], feats[:,f].shape)
      noisy_feats_2[:,f] = feats[:,f] + np.random.normal(mu, sigma[f], feats[:,f].shape)

    bins_11_16_22.append(feats)
    bins_11_16_22.append(noisy_feats_1)
    bins_11_16_22.append(noisy_feats_2)

    labels_11_16_22.append(label)
    labels_11_16_22.append(label)
    labels_11_16_22.append(label)
  else:
    continue
print("BINNED 11-16-22")

bins_11_14_22 = np.asarray(bins_11_14_22,dtype=object)
bins_11_15_22 = np.asarray(bins_11_15_22,dtype=object)
bins_11_16_22 = np.asarray(bins_11_16_22,dtype=object)

labels_11_14_22 = np.asarray(labels_11_14_22)
labels_11_15_22 = np.asarray(labels_11_15_22)
labels_11_16_22 = np.asarray(labels_11_16_22)

print(bins_11_14_22.shape)
print(bins_11_15_22.shape)
print(bins_11_16_22.shape)

X = np.vstack((bins_11_14_22, bins_11_15_22, bins_11_16_22))
y = np.hstack((labels_11_14_22, labels_11_15_22, labels_11_16_22))
print("CREATED X Y STACK")

#SAVE RAM BY DEL
del feats
del labels
del noisy_feats_1
del noisy_feats_2
del bins_11_14_22
del bins_11_15_22
del bins_11_16_22
del labels_11_14_22
del labels_11_15_22
del labels_11_16_22
del data_11_14_22
del data_11_15_22
del data_11_16_22
X = X.astype('float32')
print("CLEAR OUT SOME SPACE TO SAVE RAM MEM")

#NORMALIZE DATA
##X[:,:,0] = X[:,:,0]/np.max(np.abs(X[:,:,0])) #motion (0-5)
##X[:,:,1] = X[:,:,1]/np.max(np.abs(X[:,:,1]))
##X[:,:,2] = X[:,:,2]/np.max(np.abs(X[:,:,2]))
##X[:,:,3] = X[:,:,3]/np.max(np.abs(X[:,:,3]))
##X[:,:,4] = X[:,:,4]/np.max(np.abs(X[:,:,4]))
##X[:,:,5] = X[:,:,5]/np.max(np.abs(X[:,:,5]))
##X[:,:,6] = X[:,:,6]/np.max(np.abs(X[:,:,6])) #audio
X[:,:,0] = X[:,:,0]/np.max(np.abs(X[:,:,0]))
print("NORMALIZED")


#ONEHOT ENCODE LABELS
y = np.reshape(y, (len(y), 1))
enc = OneHotEncoder(handle_unknown='ignore')
y = enc.fit_transform(y).toarray()#one hot condaifnaosfa
print("ONEHOT ENCODING COMPLETE")


tend = t.time()
print("RUNTIME: %.2f s"%(tend-tstart))


#SAVED PREPROCESSED DATA
np.save("./P2_A_training_data.npy", arr=X)
np.save("./P2_A_labels.npy", arr=y)
print("SAVED PREPROCESSED DATA")





"""
accel_x = motion[0][:,1]/np.max(np.abs(motion[0][:,1]))
accel_y = motion[0][:,2]/np.max(np.abs(motion[0][:,2]))
accel_z = motion[0][:,3]/np.max(np.abs(motion[0][:,3]))
gyros_x = motion[0][:,4]/np.max(np.abs(motion[0][:,4]))
gyros_y = motion[0][:,5]/np.max(np.abs(motion[0][:,5]))
gyros_z = motion[0][:,6]/np.max(np.abs(motion[0][:,6]))
sound = sound[0]/np.max(np.abs(sound[0]))


plt.plot(motion_time, accel_x, label="Accel_x")
plt.plot(motion_time, accel_y, label="Accel_y")
plt.plot(motion_time, accel_z, label="Accel_z")
plt.plot(motion_time, gyros_x, label="Gyros_x")
plt.plot(motion_time, gyros_y, label="Gyros_y")
plt.plot(motion_time, gyros_z, label="Gyros_z")

print(label[0])


plt.vlines(x=label[0][:,1], ymin=0, ymax=np.max(accel_x), colors='green', ls='--', lw=1, label='Bike label 11-14(SHIFTED)')
plt.vlines(x=label[0][:,2], ymin=0, ymax=np.max(accel_x), colors='red', ls='--', lw=1, label='Bike label 11-14(SHIFTED)')

plt.title("Raw Sound Data for 11-14-22, Resampled, Smoothed, and Normalized with Events")
plt.xlabel("time [s]")
plt.legend(loc='upper left')
plt.grid()
plt.savefig("./motion_resampled_smoothed_with_events_normalized.png")





print(sound_time)
print(sound[0])
print(np.max(sound[0].to_numpy()))
plt.plot(sound_time, sound)

plt.vlines(x=label[0][:,1], ymin=0, ymax=np.max(sound[0].to_numpy()), colors='green', ls='--', lw=1, label='Bike label 11-14(SHIFTED)')
plt.vlines(x=label[0][:,2], ymin=0, ymax=np.max(sound[0].to_numpy()), colors='red', ls='--', lw=1, label='Bike label 11-14(SHIFTED)')

plt.title("Raw Sound Data for 11-14-22, Resampled, Smoothed, and Normalized with Events")
plt.xlabel('time [s]')
plt.grid()
plt.savefig("./sound_resampled_smooth_with_events_normalized.png")




#split into test/train sets
X_train, X_test, y_train, y_test = train_test_split(X,  y, test_size = 0.3, random_state = 42)

#print("train/test data")
#print(X_train.shape)
#print(X_test.shape)
#print(y_train.shape)
#print(y_test.shape)

tend = t.time()
print("Preprocessing Runtime: %.2f s"%(tend-tstart))

#print(np.unique(y_train))
#print(np.unique(y_test))

#onehot encode labels

#onehot encode training labels
y_train = y_train.astype(int)
#a = y_train
label_binarizer = LabelBinarizer()
label_binarizer.fit(range(min(y_train), max(y_train)+1))
y_train = label_binarizer.transform(y_train)
#print(y_train.shape)


#onehot encode test labels
y_test = y_test.astype(int)
#a =y_test
label_binarizer = LabelBinarizer()
label_binarizer.fit(range(min(y_test), max(y_test)+1))
y_test = label_binarizer.transform(y_test)
#print(y_test.shape)



np.save("./X_train.npy", arr=X_train)
np.save("./X_test.npy", arr=X_test)
np.save("./y_train.npy", arr=y_train)
np.save("./y_test.npy", arr=y_test)

"""
