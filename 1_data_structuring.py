import numpy as np
import pandas as pd
import scipy.io.wavfile as wavfile
import time as t
import os

tstart = t.time()

filepath = "SensorLogger/"
dir_list = os.listdir(filepath)
end = len(dir_list)
dir_list = dir_list[end-3:end]
dir_list = [dir_list[1], dir_list[2], dir_list[0]]
dir_list.sort()
print(dir_list)

#READ IN MOTION DATA
print("READ IN MOTION DATA")
motion = []
for day in dir_list:
    path_acce = filepath + day + "/"+ "Accelerometer.csv"
    path_gyro = filepath + day + "/"+"Gyroscope.csv"

    df1 = pd.read_csv(path_acce)
    df2 = pd.read_csv(path_gyro)

    df1.rename(columns={"x": "Accel_x", "y": "Accel_y","z":"Accel_z"}, inplace=True)
    df2.rename(columns={"x": "Gyros_x", "y": "Gyros_y","z":"Gyros_z"}, inplace=True)
    df1 = df1[['time', 'seconds_elapsed', 'Accel_x','Accel_y','Accel_z']]
    df2 = df2[['time', 'seconds_elapsed', 'Gyros_x','Gyros_y','Gyros_z']]

    df = pd.merge(df1, df2, on=["time","seconds_elapsed"])
    df = df.drop(["time"], axis=1)
    df.rename(columns={"seconds_elapsed": "t0"}, inplace=True)

    #motion[i] = df
    motion.append(df)
print(motion[0])

#READ IN SOUND DATA
print("READ IN SOUND DATA")
sound = []
for day in dir_list:
  mic_path = filepath + day + "/Microphone.wav"
  #samplerate, sound[i] = wavfile.read(mic_path)
  samplerate, data = wavfile.read(mic_path)
  data = pd.DataFrame(data)
  data.rename(columns={0:'sound_amplitude'}, inplace=True)
  sound.append(pd.DataFrame(data))
print(sound[0])

#READ IN EVENTS DATA
print("READ IN EVENTS DATA")
events = pd.read_csv("BikeLabels2.csv")
events_arr = []
events_arr.append(events[events["video"]=="11-14-22"])
events_arr.append(events[events["video"]=="11-15-22"])
events_arr.append(events[events["video"]=="11-16-22"])
events = events_arr
print(events[0])

#READ IN OFFSET DATA
print("READ IN OFFSET DATA")
offset = pd.read_csv("time_shift.csv")
print(offset)

#APPLY OFFSET
print("APPLY OFFSET")
for d in np.arange(len(events)):
  #print(d)
  events[d]["s_video_start"] = events[d]["s_video_start"] + offset.at[d, "s_to_senser_from_v"];
  events[d]["s_video_end"] = events[d]["s_video_end"] + offset.at[d, "s_to_senser_from_v"];
print(events[0])

#RESAMPLING FUNCTION
def resample(sound, motion, scalar):
    temp_motion = motion
    temp_motion.set_index('t0',inplace =True)
    length = int(len(sound)* (1/scalar))
    t_min = np.min(temp_motion.index)
    t_max = np.max(temp_motion.index)
    new_ts = np.linspace(t_min, t_max, length)
    resampled_motion = temp_motion.reindex(temp_motion.index.union(new_ts)).interpolate('values').loc[new_ts]
    return resampled_motion


#UPSAMPLE MOTION DATA TO 4KHZ
print("RESAMPLE MOTION DATA")
resample_motion = []
#resample_motion = {}
for i in np.arange(len(motion)):
    #resample_motion[i] = resample(sound[i], motion[i], 2)
    resample_motion.append(resample(sound[i], motion[i], 2))
motion = resample_motion
print(motion[0])

#DOWNSAMPLE SOUND DATA TO 4KHZ
print("RESAMPLE SOUND DATA")
resample_sound = []
for i in np.arange(len(sound)):
    indices = np.arange(0, len(sound[i]), 2)
    #resample_sound[i] = sound[i][indices]
    new_df = sound[i].iloc[indices]
    new_df.reset_index(drop=True, inplace=True)
    resample_sound.append(new_df)
sound = resample_sound
print(sound[0])

#APPLY ROLLING MEAN TO HELP FILTER OUT SOME NOISE
print("APPLY ROLLING MEAN")
for i in np.arange(len(motion)):
  motion[i] = motion[i].rolling(10).mean().dropna(axis=0)
  motion[i] = motion[i].reset_index()
  sound[i] = sound[i].rolling(10).mean().dropna(axis=0)
  sound[i] = sound[i].reset_index(drop=True)

#MERGE MOTION WITH SOUND
data = []
for i in np.arange(len(motion)):
    data.append(motion[i].merge(right=sound[i], how='outer', left_index=True, right_index=True))
print(data[0])

#CREATE LABELS AND SET DEFAULT TO ZERO
for i in np.arange(len(data)):
  zeros = np.zeros([len(data[i]),1])
  data[i] = np.hstack([data[i], zeros])
print(data[0])

#CONVERT EVENTS TO NUMPY
for i in np.arange(len(events)):
    events[i] = events[i].to_numpy()

#REASSIGN LABELS (+ FOR IN, - FOR OUT)
print("REASSIGN LABELS")
for d in np.arange(len(events)):
  #print(d)
  start = events[d][:,1] #start timestamp
  end = events[d][:,2] #end timestamp
  event = events[d][:,3] #in or out
  time = data[d][:,0] #timestamps from motion
  for i in np.arange(events[d].shape[0]):
    if event[i] == "In":
      data[d][:,-1] = data[d][:,-1] + ((start[i] <= time) & (end[i] >= time))
    else:
      #print(motion[d][:,7].shape)
      #print([(start[0] <= time) & (end[0] >= time)].shape)
      data[d][:,-1] = data[d][:,-1] + -1*((start[i] <= time) & (end[i] >= time))

#SAVE RESTRUCTURED DATA ARRAYS
np.save("./data_11-14-22.npy", arr=data[0])
np.save("./data_11-15-22.npy", arr=data[1])
np.save("./data_11-16-22.npy", arr=data[2])
print("SAVED NEW DATASETS")

tend = t.time()
print("RUNTIME: %.2f s"%(tend-tstart))