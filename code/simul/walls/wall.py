import numpy as np
from typing import Union


def parallel_wall_dist(x: float, t: float, wall_pos: float) -> float:

    return 2 * ((x / 2) ** 2 + (wall_pos) ** 2) ** 0.5


def normal_wall_dist(x: float, t: float, wall_pos: float) -> float:

    # return 2 * abs(wall_pos - x) + x

    # X -> wall -> receiver

    return abs(wall_pos - x) + abs(wall_pos)


# def get_signal_path(x:float, t:float)->float:

#     return abs(x)


class SignalPath:
    def __init__(self):
        pass

    def path_length(self, x: Union[float, np.ndarray], t: Union[float, np.ndarray]):

        raise NotImplementedError

    @staticmethod
    def from_dict(d: dict()) -> 'SignalPath':
        # print(d)
        if not "type" in d:
            return None

        if(d["type"] == "LOS"):
            return Path_LOS()

        if(d["type"] == "ParralelWall"):
            return Path_ParralelWall(d["wall_y"])

        if(d["type"] == "NormalWall"):
            return Path_NormalWall(d["wall_x"])

        return SignalPath()

    def to_dict(self) -> dict:

        return {"type": "SignalPath"}


class Path_LOS(SignalPath):
    def __init__(self):
        pass

    def path_length(self, x: Union[float, np.ndarray], t: Union[float, np.ndarray]):
        # print(f"LOS:{x}")
        return np.abs(x)

    def to_dict(self):

        return {"type": "LOS"}


class Path_ParralelWall(SignalPath):
    def __init__(self, wall_dist):

        self.wall_dist = wall_dist

    def path_length(self, x: Union[float, np.ndarray], t: Union[float, np.ndarray]):

        return 2 * ((x / 2) ** 2 + (self.wall_dist) ** 2) ** 0.5

    def to_dict(self):

        return {"type": "ParralelWall", "wall_y": self.wall_dist}


class Path_NormalWall(SignalPath):
    def __init__(self, wall_x):

        self.wall_x = wall_x

    def path_length(self, x: Union[float, np.ndarray], t: Union[float, np.ndarray]):

        return np.abs(x - self.wall_x) + np.abs(self.wall_x)

    def to_dict(self):

        return {"type": "NormalWall", "wall_x": self.wall_x}
