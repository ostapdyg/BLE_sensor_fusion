import pandas as pd
import numpy as np
import plotly.express as plx

from ahrs.filters import Madgwick

def euler_from_quaternion(x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z # in radians


df = pd.read_csv("imu_2.csv", sep=",")

df = pd.read_csv("HyperIMU.csv", sep=",")
df_plt = df
acc_cols = ["icm4x6xx_accelerometer.x","icm4x6xx_accelerometer.y", "icm4x6xx_accelerometer.z", "icm4x6xx_accelerometer"]
# ak0991x_magnetometer.x,ak0991x_magnetometer.y,ak0991x_magnetometer.z,icm4x6xx_gyroscope.x,icm4x6xx_gyroscope.y,icm4x6xx_gyroscope.z
mag_cols = ["ak0991x_magnetometer.x","ak0991x_magnetometer.y", "ak0991x_magnetometer.z"]
gyro_cols = ["icm4x6xx_gyroscope.x","icm4x6xx_gyroscope.y", "icm4x6xx_gyroscope.z"]
# df["time"] = np.arange(0, )
print(df.columns, df.shape)
df["icm4x6xx_accelerometer"] = (df["icm4x6xx_accelerometer.x"]**2 + df["icm4x6xx_accelerometer.y"]**2 + df["icm4x6xx_accelerometer.z"]**2)**.5
df["time"] = np.arange(df.shape[0])*0.005

from ahrs.filters import Madgwick
mad = Madgwick(gyr=np.array(df[gyro_cols]), acc=np.array(df[acc_cols]), frequency = 20)
print(mad.Q)
angles = np.array([euler_from_quaternion(*quats) for quats in mad.Q])
plx.line(x=df["time"], y=angles[:,2], markers=True).show()