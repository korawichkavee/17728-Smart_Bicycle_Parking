import itertools

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def data_gen():
    for cnt in itertools.count():
        t = cnt / 10
        yield t, np.sin(2*np.pi*t) * np.exp(-t/10.)

def init():
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(xmin, xmax)
    del xdata[:]
    del ydata[:]
    line.set_data(xdata, ydata)
    return line,

fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2)
ax.grid()
xdata, ydata = [], []

def run(data):
    # update the data
    t, y = data
    xdata.append(t)
    ydata.append(y)
    xmin, xmax = ax.get_xlim()

    if t >= xmax:
        ax.set_xlim(xmin, 2*xmax)
        ax.figure.canvas.draw()
    line.set_data(xdata, ydata)

    return line,
###############
import os
import pandas as pd
# get a list of all folders
filepath = "SensorLogger/"
dir_list = os.listdir(filepath)
dir_list.sort()
n = 0
#read csv file for 2 sensors
path_acce = filepath + "/"+ dir_list[n] + "/"+ "Accelerometer.csv"
path_gyro = filepath + "/"+ dir_list[n] + "/"+"Gyroscope.csv"
df1 = pd.read_csv(path_acce)
print(df1.shape)
df2 = pd.read_csv(path_gyro)
print(df2.shape)
#rename col
df1.rename(columns={"x": "Accel_x", "y": "Accel_y","z":"Accel_z"}, inplace=True)
df2.rename(columns={"x": "Gyros_x", "y": "Gyros_y","z":"Gyros_z"}, inplace=True)
df1 = df1[['time', 'seconds_elapsed', 'Accel_x','Accel_y','Accel_z']]
df2 = df2[['time', 'seconds_elapsed', 'Gyros_x','Gyros_y','Gyros_z']]
df = pd.merge(df1, df2, on=["time","seconds_elapsed"])
df = df.drop(["time"], axis=1)
#df["seconds_elapsed"] = df["seconds_elapsed"]*1000
df.rename(columns={"seconds_elapsed": "t0"}, inplace=True)
print(df.shape)

##print("READ IN SOUND DATA")
##sound = []
##for day in dir_list:
##  mic_path = filepath + day + "/Microphone.wav"
##  #samplerate, sound[i] = wavfile.read(mic_path)
##  samplerate, data = wavfile.read(mic_path)
##  data = pd.DataFrame(data)
##  data.rename(columns={0:'sound_amplitude'}, inplace=True)
##  sound.append(pd.DataFrame(data))

xmin = 681-5+611.930007
xmax = 720+5+611.930007
df = df[df['t0']>=xmin]
df = df[df['t0']<=xmax]


ymin = df['Accel_z'].min()-0.25*abs(df['Accel_z'].min())
ymax = df['Accel_z'].max()+0.25*abs(df['Accel_z'].max())
###############
##ani = animation.FuncAnimation(fig, run, data_gen, interval=10, init_func=init)
ani = animation.FuncAnimation(fig, run, df[['t0','Accel_z']].to_numpy(), interval=10, init_func=init )
print("ANIMATION IS SET")
##plt.show()
##ani.save('increasingStraightLine.avi', fps=60, dpi=500)
ani.save('sensorlogger_14_681.avi', fps=80, dpi=500)
print("VIDEO SAVED")
##plt.close()