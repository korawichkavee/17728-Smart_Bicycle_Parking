import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import time as t

tstart = t.time()

dir_list = ["11-14-22", "11-15-22", "11-16-22"]

#LOAD ALL PREVIOUS OUTPUTS
y_true = np.load("./P2_M_y_true.npy")
y_pred = np.load("./P2_M_y_pred.npy")

training_performance = np.load("./P2_M_ML_Project_performance.npy", allow_pickle=True)
epochs = len(training_performance[()]['val_accuracy'])

print("y_true")
print(y_true)
print("y_pred")
print(y_pred)
"""
raw_data = []
for day in dir_list:
    fig, ax = plt.subplots(3)
    fp_in = "./data_" + day + ".npy"
    data = np.load(fp_in)

    time = data[:,0]
    x_acc = data[:,1]
    y_acc = data[:,2]
    z_acc = data[:,3]
    x_gyro = data[:,4]
    y_gyro = data[:,5]
    z_gyro = data[:,6]
    sound = data[:,7]

    ax[0].plot(time, x_acc, label='x_acc')
    ax[0].plot(time, y_acc, label='y_acc')
    ax[0].plot(time, z_acc, label='z_acc')
    ax[0].grid()
    ax[0].legend(loc='upper right')
    ax[0].set_title("Raw Accelerometer")
    ax[0].set_xlabel("Seconds")
    ax[0].set_ylabel("Acceleration")

    ax[1].plot(time, x_acc, label='x_gyro')
    ax[1].plot(time, y_acc, label='y_gyro')
    ax[1].plot(time, z_acc, label='z_gyro')
    ax[1].grid()
    ax[1].legend(loc='upper right')
    ax[1].set_title("Raw Gyroscope")
    ax[1].set_xlabel("Seconds")
    ax[1].set_ylabel("Gyroscope???")

    ax[2].plot(time, sound, label='sound')
    ax[2].grid()
    ax[2].legend(loc='upper right')
    ax[2].set_title("Raw Sound")
    ax[2].set_xlabel("Seconds")
    ax[2].set_ylabel("Sound Amplitude??")

    fig.tight_layout()
    fp_out = "./raw_" + day + ".png"
    fig.savefig(fp_out)
"""







#breakpoint()


#CONFUSION MATRIX FOR TEST DATA
cm = confusion_matrix(y_true, y_pred )#, normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[-1,1])
##disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[-1,1])
disp.plot()
plt.savefig("./P2_M_confusion_matrix_test.png")
print("CONFUSION MATRIX SAVED")


#TRAINING AND VALIDATION [LOSS]
fig_loss, ax_loss = plt.subplots()
ax_loss.plot(np.arange(epochs)+1, training_performance[()]['loss'],label='categorical cross entropy loss (train)')
ax_loss.plot(np.arange(epochs)+1, training_performance[()]['val_loss'], label='categorical cross entropy loss (val)')
ax_loss.set_title('MSE loss performance over %d epochs'%(epochs))
ax_loss.set_xlabel('# of epochs')
ax_loss.legend(loc='upper right')
fig_loss.savefig("./P2_M_performance_loss.png")
print("TRAININD AND VALIDATION LOSS SAVED")

#TRAINING AND VALIDATION [ACCURACY]
fig_accuracy, ax_accuracy = plt.subplots()
ax_accuracy.plot(np.arange(epochs)+1, training_performance[()]['accuracy'],label='accuracy (train)')
ax_accuracy.plot(np.arange(epochs)+1, training_performance[()]['val_accuracy'], label='accuracy (val)')
ax_accuracy.set_title('Accuracy performance over %d epochs'%(epochs))
ax_accuracy.set_xlabel('# of epochs')
ax_accuracy.legend(loc='upper right')
fig_accuracy.savefig("./P2_M_performance_accuracy.png")
print("TRAININD AND VALIDATION ACCURACY SAVED")
