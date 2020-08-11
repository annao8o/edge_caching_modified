import numpy as np
import matplotlib.pyplot as plt
import random
import networkx as nx
import copy

class Cloud:
    def __init__(self):
        self.library = None
        self.server_num = None
        self.contents_num = None
        self.server_lst = list()
        self.graph = None
        self.cluster_num = None

    def set_env(self, server_num, contents_num, cluster_num):
        self.server_num = server_num
        self.contents_num = contents_num
        self.cluster_num = cluster_num

    def create_env(self, data_lst):
        g = nx.Graph()

        for i in range(self.server_num):
            s = EdgeServer(i)
            self.server_lst.append(s)
            g.add_node(i)
            print("Success to add server")

        self.graph = g

    def make_cluster(self, users):
        for i in range(self.cluster_num[0], self.cluster_num[1]):
            pass


    def asso_server_to_cluster(self):
        for i in self.server_lst:
            # i.cluster =  ### #어떤 기준으로 associate 할 것인지?
            pass

class EdgeServer:
    def __init__(self, id):
        self.id = id
        self.state = None
        self.content = None
        self.cluster = None

    def asso_cluster(self):
        pass


class User:
    def __init__(self):
        self.pref_vec = None

    def make_pref(self, z_val, contents_num):
        z = Zipf()
        z.set_env(z_val, contents_num)
        self.pref_vec = copy.copy(z.pdf)


class PPP:
    def __init__(self):
        #simulation window parameters
        self.lambda0 = None
        self.xx = None
        self.yy = None

    def set_env(self, density):
        #rectangle dimensions
        r = 1
        xx0 = 0
        yy0 = 0
        areaTotal = np.pi * r ** 2

        #point process parameters
        self.lambda0 = density

        #simulate poisson point process

        numbPoints = np.random.poisson(self.lambda0*areaTotal)   #poisson number of points
        theta = 2 * np.pi * np.random.uniform(0, 1, numbPoints)
        rho = r * np.sqrt(np.random.uniform(0, 1, numbPoints))

        self.xx = rho * np.cos(theta)
        self.yy = rho * np.sin(theta)

        self.xx = self.xx + xx0
        self.yy = self.yy + yy0


class Zipf:
    def __init__(self):
        self.pdf = None

    def set_env(self, expn, num_contents):
        temp = np.power(np.arange(1, num_contents+1), -expn)
        zeta = np.r_[0.0, np.cumsum(temp)]
        self.pdf = [x / zeta[-1] for x in zeta]

    def get_sample(self):
        f = random.random()
        return np.searchsorted(self.pdf, f) - 1


# if __name__ == '__main__':
#     server = PPP()
#     server.set_env(5)
#
#     user = PPP()
#     user.set_env(20)
#
#     plt.scatter(server.xx, server.yy, edgecolors='b', facecolor='b', alpha=0.5)
#     plt.scatter(user.xx, user.yy, edgecolors='r', facecolor='r', alpha=0.5)
#
#     plt.xlabel('x')
#     plt.ylabel('y')
#     plt.show()