import torch
import numpy as np
import math
import matplotlib.pyplot as plt
import pickle as pkl
from train_data_gen import grid2D, data_gen

def zero_isocontour(a,b,p,theta):
    r = a * math.cos(p * theta) + b
    dr = - a * p * math.sin(p * theta)
    ddr = - a * (p ** 2) * math.cos(p * theta)
    return r, dr, ddr

def three_pedal(r, a, b, p, theta):
    return r - a * math.cos(p * theta) - b

def analytic_curvature(r, dr, ddr):
    return ((r ** 2) + 2 * (dr ** 2) - r * ddr) / (((r ** 2) + (dr ** 2)) ** (3/2))



if __name__ == '__main__':
    xmin = -0.207843
    xmax = 0.207843
    ymin = -0.207843
    ymax = 0.207843

    a = 0.05 # 0.05 for smooth flower and 0.075 for sharp flower
    b = 0.15
    p = 3

    x0 = 0.0
    y0 = 0.0

    dataset = []
    label = []

    N = 107

    solution = torch.zeros(N*N)

    for i in range(N*N):
        x = data_gen_.grid.x_from_n(i)
        y = data_gen_.grid.y_from_n(i)

        r = sqrt(x ** 2 + y ** 2)
        theta = math.atan2(y,x)

        solution[i] = three_pedal(r, a,b,p, theta)

    
