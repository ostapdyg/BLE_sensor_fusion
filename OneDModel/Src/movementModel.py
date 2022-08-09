import numpy as np


class MovementModel:
    def __init__(self):
        self.x = np.array((0.0, 0.0))
        self.speed = 1.4  # m/s
        self.vel = np.array((0.0, 0.0))
        # self.waypoinnts = {
        #     0:np.array((0, 0)),
        #     1:np.array((4.8, 0)),
        #     2:np.array((4.8, 2.4)),
        #     3:np.array((4.8, 2.4)),
        #     3:np.array((0, 0))
        # }
        self.waypoints = {  # time:vel
            0: np.array((1.4, 0)),
            3: np.array((0, 1.4)),
            5: np.array((0, 0)),
            6: np.array((0, -1.4)),
            8: np.array((-1.4, 0))
        }

    def update(self, t, dt):
        self.x += dt*self.vel
        if (t in self.waypoints):
            self.vel = self.waypoints[t]
        return self.x
