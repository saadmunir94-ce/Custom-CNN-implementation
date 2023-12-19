import numpy as np


class Celsius:
    def __init__(self, temp=0):
        self._temp_orig = temp
        self.n = None

    @property
    def temp(self):
        print("Temp property called")
        return self._temp_orig

    @temp.setter
    def temp(self, val):

        if val < -273:
            raise ValueError("Value <-273")
        self._temp_orig = val
        print("Temperature set")

    @property
    def temperature(self):

        if self.temp > 100:
            return self._temp_orig


cel = Celsius(300)
print(cel.temperature)
