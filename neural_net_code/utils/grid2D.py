import torch
import numpy as np
import math

class grid2D():
    def __init__(self, xmin, xmax, ymin, ymax, N, M = -1):
        if M == -1:
            self.M = N
        else:
            self.M = M

        self.N = N

        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

        self.dx = (xmax - xmin) / np.double(N - 1)
        self.dy = (ymax - ymin) / np.double(M - 1)

    def x_from_n(self, n):
        return self.xmin + (n%self.N) * self.dx

    def y_from_n(self, n):
        return self.ymin + np.double(np.floor(n/self.N) * self.dx)

    def n_from_ij(self, i ,j):
        return i + j * self.N
