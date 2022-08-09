import pandas as pd
import numpy as np
import plotly.express as plx
from scipy import signal

import math

sample_rate = 0.005

# df = pd.read_csv("HyperIMU.csv", sep=",")
# df = pd.read_csv("HyperIMU2.csv", sep=",")
df = pd.read_csv("HyperIMU3.csv", sep=",")
# df = pd.read_csv("HyperIMU5.csv", sep=",")
# df = pd.read_csv("HyperIMU_fast.csv", sep=",")


df_plt = df
acc_cols = ["icm4x6xx_accelerometer.x",
            "icm4x6xx_accelerometer.y", "icm4x6xx_accelerometer.z"]
mag_cols = ["ak0991x_magnetometer.x",
            "ak0991x_magnetometer.y", "ak0991x_magnetometer.z"]
gyro_cols = ["icm4x6xx_gyroscope.x",
             "icm4x6xx_gyroscope.y", "icm4x6xx_gyroscope.z"]
df[gyro_cols] *= (180 / np.pi)

df["acc_total"] = (df["icm4x6xx_accelerometer.x"]**2 +
                   df["icm4x6xx_accelerometer.y"]**2 + df["icm4x6xx_accelerometer.z"]**2)**.5
df["rot_total"] = (df["icm4x6xx_gyroscope.x"]**2 +
                   df["icm4x6xx_gyroscope.y"]**2 + df["icm4x6xx_accelerometer.z"]**2)**.5

df["g"] = np.ones(df.shape[0])*9.8

df["time"] = np.arange(df.shape[0])*sample_rate

df["speed"] = np.zeros(df.shape[0])
print(df.shape)
speed = 0
for i in range(df.shape[0]):
    speed += (df["icm4x6xx_accelerometer.y"][i] - 5.5)*sample_rate
    df["speed"][i] = speed
# plx.line(df_plt,  x="time", y=acc_cols +
#          ["acc_total", "g"], markers=False).show()


for acc in acc_cols+["acc_total"]:
    b, a = signal.butter(5, 0.3)
    df[acc] = signal.lfilter(b, a, df[acc])

speed = -0.8
for i in range(df.shape[0]):
    speed += (df["icm4x6xx_accelerometer.y"][i] -
              5.5)*sample_rate
    df["speed"][i] = speed
plx.line(df_plt, x="time", y=acc_cols +
         ["acc_total", "g"], markers=False).show()


df["rot"] = np.zeros(df.shape[0])
rot = 0
for i in range(df.shape[0]):
    if (np.abs(df["icm4x6xx_gyroscope.z"][i]) >= 30):
        rot += df["icm4x6xx_gyroscope.z"][i]*sample_rate
    df["rot"][i] = rot
# plx.line(df_plt, x="time", y=gyro_cols + ["rot"], markers=False).show()

for col in gyro_cols:
    b, a = signal.butter(5, 0.3)
    df[col] = signal.lfilter(
        b, a, df[col])

rot = 0
for i in range(df.shape[0]):
    if (np.abs(df["icm4x6xx_gyroscope.z"][i]) >= 20):
        rot += df["icm4x6xx_gyroscope.z"][i]*sample_rate
    df["rot"][i] = rot
plx.line(df_plt, x="time", y=gyro_cols + ["rot"], markers=False).show()


df_acc = df[["acc_total", "time"]]

# plx.line(df_acc, x="time", y= ["acc_total"], markers=True).show()

# b, a = signal.butter(5, 0.3)
# df_acc["acc_filt"] = signal.filtfilt(
#             b, a, df_acc["acc_total"])

# plx.line(df_acc, x="time", y= ["acc_total", "acc_filt"], markers=True).show()
