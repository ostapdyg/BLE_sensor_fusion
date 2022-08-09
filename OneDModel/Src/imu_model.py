import numpy as np
import pandas as pd

from scipy import signal

import math


def read_dataset(f):
    df = pd.read_csv(f, sep=",")
    sample_rate = 0.05
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
    df["time"] = np.arange(df.shape[0])*sample_rate

    for col in acc_cols+["acc_total"] + gyro_cols:
        b, a = signal.butter(5, 0.3)
        df[col] = signal.filtfilt(b, a, df[col])

    return df


class IMUModel:
    def __init__(self, f):
        self.data = read_dataset(f)
        # self.t = 0
        self.idx = 0
        self.cur_step_speed = 0
        self.prev_speed = 0
        self.rot = 0
        self.sample_rate = 0.05
        self.t_step_start = 0
        self.last_step_duration = 0.5
        self.acc_prev = 0
        self.acc_max = 0
        self.step_len = 1


    def update_rot(self):
        dr = self.data["icm4x6xx_gyroscope.z"][self.idx]*1.7
        self.rot += dr*self.sample_rate

    def update_step(self):
        t = self.data["time"][self.idx]
        step_duration = t - self.t_step_start
        acc = self.data["acc_total"][self.idx]
        if (acc > self.acc_max):
            self.acc_max = acc

        self.cur_step_speed = (self.step_len /
                            (step_duration + self.last_step_duration))
        # if(step_duration > self.last_step_duration*0.5):
        #     self.cur_step_speed = 

        if (acc >= 9.8) and (self.acc_prev < 9.8):  # Step detected

            print(
                f"  step detected: {step_duration:.2f} s; {self.acc_max - 9.8:.2f} m/s2")
            if ((self.acc_max - 9.8) >= 0.5) and (0.1 < step_duration):
                step_duration = step_duration if step_duration <= 1.0 else 1.0
                self.cur_step_speed = self.step_len / step_duration
                print(
                    f"    cur speed:{self.cur_step_speed:.2f}m/s; prev:{self.prev_speed:.2f}m/s")
                self.last_step_duration = step_duration
                step_duration = 0
                self.t_step_start = t
            self.acc_max = 0

        self.acc_prev = acc

    def get_rot(self):
        res = self.rot
        self.rot = 0
        return res

    def get_speed(self):
        return  self.cur_step_speed

    def update(self, t, dt):
        while (self.idx < self.data.shape[0]):
            if(self.data["time"][self.idx] > t):
                return
            self.update_rot()
            self.update_step()

            self.idx += 1
