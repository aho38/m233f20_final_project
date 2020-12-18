import pickle as pkl
import torch
from train_data_gen import grid2D, data_gen
import numpy as np
import math

def sdf_circular(x, y, x0, y0, r):
    return math.sqrt( (x - x0) ** 2 + (y - y0) ** 2 ) - r

if __name__ == '__main__':
    N = 255
    xmin = -0.5
    xmax = 0.5
    ymin = -0.5
    ymax = 0.5

    x0 = 0.0
    y0 = 0.0

    #list for storing dataset
    dataset = []
    label = []

    data_gen_ = data_gen(xmin, xmax, ymin, ymax, N)

    r_ls = np.linspace(1.6 * data_gen_.grid.dx, 0.5 - 2.0 * data_gen_.grid.dx, 100, dtype=np.double())

    theta_ls = np.linspace(0, 2*math.acos(-1), 20, dtype=np.double())

    for r in r_ls:
        for theta in theta_ls:
            node_indexs = data_gen_.get_data_pt(r, x0, y0, theta)

            # ==========================
            data_tensor = torch.zeros(9)
            for j in range(9):
                x = data_gen_.grid.x_from_n(node_indexs[j])
                y = data_gen_.grid.y_from_n(node_indexs[j])

                # print('x:', x)
                # print('y:', y)
                # print('r:', r)

                data_tensor[j] = sdf_circular(x, y, x0, y0, r)

            # print('data tensor:', data_tensor)
            dataset.append(data_tensor)
            label.append(torch.tensor([data_gen_.grid.dx / r]))

            dataset.append(- data_tensor)
            label.append(torch.tensor([- data_gen_.grid.dx / r]))

    pkl.dump(torch.stack(dataset, 0), open('./data/test_circle/'+ str(N)+'_data.pkl', 'wb'))
    pkl.dump(torch.stack(label, 0), open('./data/test_circle/' + str(N)+ '_label.pkl', 'wb'))
