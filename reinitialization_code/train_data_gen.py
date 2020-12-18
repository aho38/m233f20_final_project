import torch
import numpy as np
import math
import matplotlib.pyplot as plt
import pickle

def sdf_circular(x, y, x0, y0, r):
    return math.sqrt( (x - x0) ** 2 + (y - y0) ** 2 ) - r

def nsdf_circular(x, y, x0, y0, r):
    return (x - x0) ** 2 + (y - y0) ** 2 - r ** 2

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
        return self.xmin + np.double((n%self.N) * self.dx)

    def y_from_n(self, n):
        return self.ymin + np.double(np.floor(n/self.N) * self.dx)

    def n_from_ij(self, i ,j):
        return i + j * self.N

class data_gen():
    def __init__(self, xmin, xmax, ymin, ymax, N, M = -1):
        self.grid = grid2D(xmin, xmax, ymin, ymax, N, N)

    def get_node(self, r, x0, y0, theta):
        # binary 1 = use ceil; 0 = use floor
        x_ceil_selec = np.random.randint(0, 2)
        y_ceil_selec = np.random.randint(0, 2)

        # x and y val that is on the 0-contour
        x = r * math.cos(theta) + x0
        y = r * math.sin(theta) + y0

        # i and j val on the grid
        i = (x - self.grid.xmin) / self.grid.dx
        j = (y - self.grid.ymin) / self.grid.dy

        #randomly select the next node or the previous
        if (i - math.floor(i)) < 0.5:
            i = math.floor(i)
        else:
            i = math.ceil(i)
        if (j - math.floor(j)) < 0.5:
            j = math.floor(j)
        else:
            j = math.ceil(j)

        return self.grid.n_from_ij(i, j)

    def get_data_pt(self, r, x0, y0, theta):
        ij_node = self.get_node(r, x0, y0, theta)

        #get the 9 node points you need for data starting from top left as n1 and go right
        n1 = ij_node + self.grid.N - 1
        n2 = ij_node + self.grid.N
        n3 = ij_node + self.grid.N + 1
        n4 = ij_node - 1
        n5 = ij_node
        n6 = ij_node + 1
        n7 = ij_node - self.grid.N - 1
        n8 = ij_node - self.grid.N
        n9 = ij_node - self.grid.N + 1

        tensor_nodes = torch.Tensor([n1, n2, n3, n4, n5, n6, n7, n8, n9])

        if tensor_nodes[tensor_nodes<0].nelement() != 0:
            raise Exception("You are out of bound!")

        return tensor_nodes

if __name__ == "__main__":
    xmin = 0
    xmax = 1
    ymin = 0
    ymax = 1

    #list for storing dataset
    dataset = []
    label = []

    N = 266
    v = np.int(np.ceil((N - 8.2) / 2 + 1))
    #print(v)

    data_gen_ = data_gen(xmin, xmax, ymin, ymax, N)

    # get a random x0 and y0 between (x0, y0) in [0.5 - h / 2 , 0.5 + h / 2]
    x0 = np.random.uniform(0.5 - data_gen_.grid.dx / 2.0, 0.5 + data_gen_.grid.dx / 2.0)
    y0 = np.random.uniform(0.5 - data_gen_.grid.dx / 2.0, 0.5 + data_gen_.grid.dx / 2.0)
    r_ls = np.linspace(1.6 * data_gen_.grid.dx, 0.5 - 2.0 * data_gen_.grid.dx, v )
    theta_ls = np.linspace(0, 2*math.acos(-1), 360 * 2)
    print(len(r_ls))

    for repeat in range(5):
        for r in r_ls:
            for theta in theta_ls:
                node_indexs = data_gen_.get_data_pt(r, x0, y0, theta)

                # ==========================
                data_tensor = torch.zeros(9)
                for j in range(9):
                    x = data_gen_.grid.x_from_n(node_indexs[j])
                    y = data_gen_.grid.y_from_n(node_indexs[j])

                    data_tensor[j] = sdf_circular(x, y, x0, y0, r)

                dataset.append(data_tensor)
                label.append(torch.tensor([data_gen_.grid.dx / r]))

                dataset.append(- data_tensor)
                label.append(torch.tensor([- data_gen_.grid.dx / r]))
                # ==========================
                for j in range(9):
                    x = data_gen_.grid.x_from_n(node_indexs[j])
                    y = data_gen_.grid.y_from_n(node_indexs[j])

                    data_tensor[j] = nsdf_circular(x, y, x0, y0, r)

                dataset.append(data_tensor)
                label.append(torch.tensor([data_gen_.grid.dx / r]))

                dataset.append(- data_tensor)
                label.append(torch.tensor([- data_gen_.grid.dx / r]))

    pickle.dump(torch.stack(dataset, 0), open('./data/'+str(N)+'_data.pkl','wb'))
    pickle.dump(torch.stack(label, 0), open('./data/'+str(N)+'_label.pkl','wb'))
