import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import random
import networkx as nx
import copy
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import math
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(0)
class Cloud:
    def __init__(self):
        self.library = None
        self.server_num = None
        self.contents_num = None
        self.server_lst = list()
        self.graph = None
        self.users = list()
        self.cluster_num = None

    def set_env(self, server_num, contents_num, cluster_num):
        self.server_num = server_num
        self.contents_num = contents_num
        self.cluster_num = cluster_num
        self.create_env()

    def create_env(self):
        g = nx.Graph()

        for i in range(self.server_num):
            s = EdgeServer(i)
            self.server_lst.append(s)
            g.add_node(i)
            print("Success to add server")

        self.graph = g
        self.library = np.arange(1, self.contents_num+1)

    def add_user(self, user):
        self.users.append(user)

    def make_cluster(self):
        p_vectors = list()
        for user in self.users:
            p_vectors.append(user.pref_vec)

        # for i in range(self.cluster_num[0], self.cluster_num[1]):
        #n_cluster = int(self.cluster_num[0])
        n_cluster = 3


        ## dimension reduction (PCA) for visualization
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(p_vectors)

        kmeans = KMeans(n_clusters=n_cluster, init='random', algorithm='auto')
        kmeans.fit(pca_data)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(pca_data[:, 0], pca_data[:, 1], c=kmeans.labels_.astype(float), alpha=0.5)
        for i in range(len(pca_data)):
            ax.annotate(str(i), xy=(pca_data[i][0], pca_data[i][1]))
        plt.show()

        return kmeans.labels_

    def asso_server_to_cluster(self):
        for i in self.server_lst:
            # i.cluster =  ### #어떤 기준으로 associate 할 것인지?
            pass

    def cal_cluster_p_vec(self):
        pass

class EdgeServer:
    def __init__(self, id):
        self.id = id
        self.state = None
        self.content = None
        self.cluster = None
        self.total_request = 0
        self.algo_lst = list()

    def add_algo(self, algo):
        if type(algo).__name__ == 'CacheAlgo':
            self.algo_lst.append(algo)
            print('Success to add algo')
        else:
            print('wrong algo class')

    def asso_cluster(self):
        pass

    def request_content(self, content_id):
        print('requests content (id: {})'.format(content_id))

        self.total_request += 1
        hit_lst = list()

        for algo in self.algo_lst:
            hit = algo.have_content(content_id)
            if hit:
                print("{} >> hit!".format(algo.id))
                hit_lst.append(1)
            else:
                print("{} >> no hit!".format(algo.id))
                hit_lst.append(0)
                algo.replacement_content(content_id)

        return hit_lst


class User:
    def __init__(self, id):
        self.pref_vec = list()
        self.id = id

    def make_pref(self, z_val, contents_num):
        self.pref_vec = [0 for _ in range(contents_num)]

        z = Zipf()
        z.set_env(z_val, contents_num)
        idx_lst = np.arange(0, contents_num)
        np.random.shuffle(idx_lst)
        tmp = copy.copy(z.pdf)

        for i in range(contents_num):
            self.pref_vec[idx_lst[i]] = tmp[i]

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
        theta = 2 * np.pi * np.random.uniform(0, 1, numbPoints)  #angular coordinates
        rho = r * np.sqrt(np.random.uniform(0, 1, numbPoints))   #radial coordinates

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
        self.pdf = [x / zeta[-1] for x in temp]

    def get_sample(self):
        f = random.random()
        return np.searchsorted(self.pdf, f) - 1


if __name__ == '__main__':
    # server = PPP()
    # server.set_env(5)
    #
    # user = PPP()
    # user.set_env(20)
    #
    # plt.scatter(server.xx, server.yy, edgecolors='b', facecolor='b', alpha=0.5)
    # plt.scatter(user.xx, user.yy, edgecolors='r', facecolor='r', alpha=0.5)
    #
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.show()

    user1 = User()
    user1.make_pref(0.7, 10)
    print(user1.pref_vec)

    user2 = User()
    user2.make_pref(0.7, 10)
    print(user2.pref_vec)