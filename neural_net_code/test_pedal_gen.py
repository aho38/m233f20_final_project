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

    data_gen_ = data_gen(xmin, xmax, ymin, ymax, N)

    for i in range(552):
        data_tensor = torch.zeros(9)
        k = 0

        theta = np.random.uniform(0,2 * math.acos(-1))
        r, dr, ddr = zero_isocontour(a,b,p,theta)

        node_indexs = data_gen_.get_data_pt(r, x0, y0, theta)

        for j in range(9):
            x = data_gen_.grid.x_from_n(node_indexs[j])
            y = data_gen_.grid.y_from_n(node_indexs[j])

            n_r = math.sqrt(x ** 2 + y ** 2)
            n_theta = math.atan2(y,x)

            data_tensor[j] = three_pedal(n_r, a,b,p,n_theta)

        print(data_tensor)

        k = analytic_curvature(r, dr, ddr)
        hk = k * data_gen_.grid.dx

        dataset.append(data_tensor)
        label.append(torch.tensor([hk]))

    # pkl.dump(torch.stack(dataset, 0), open('./data/test_pedal/255_data.pkl', 'wb'))
    # pkl.dump(torch.stack(label, 0), open('./data/test_pedal/255_label.pkl', 'wb'))
