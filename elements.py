import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import random
import networkx as nx
import copy
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from prediction_model import *
import math
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(0)
random.seed(0)
class Cloud:
    def __init__(self):
        self.library = None
        self.server_position = None
        self.user_position = None
        self.contents_num = None
        self.server_lst = list()
        self.cluster_lst = list()
        self.graph = None
        self.user_lst = list()
        self.cluster_num = None
        self.m = None

    def set_env(self, server_position, user_position, contents_num, cluster_num, model):
        self.server_position = server_position
        self.user_position = user_position
        self.contents_num = contents_num
        self.cluster_num = cluster_num
        self.m = model
        self.create_env()

    def create_env(self):
        g = nx.Graph()

        for i in range(len(self.server_position)):
            s = EdgeServer(i, self.server_position[i])
            self.server_lst.append(s)
            g.add_node(i)
            print("Success to add server")

        for i in range(len(self.user_position)):
            u = User(i, self.user_position[i])
            self.user_lst.append(u)

        for i in range(self.cluster_num):
            cluster = Cluster(i)
            self.cluster_lst.append(cluster)

        self.graph = g
        self.library = np.arange(1, self.contents_num+1)


    def training(self, learning_rate, num_epochs, train_loader, val_loader):
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.m.parameters(), lr=learning_rate)

        trn_loss_list = []
        val_loss_list = []

        for epoch in range(num_epochs):
            trn_loss = 0.0
            for i, data in enumerate(train_loader):
                x, label = data
                if is_cuda:
                    x = x.cuda()
                    label = label.cuda()
                # grad init
                optimizer.zero_grad()
                # forward propagation
                model_output = self.m(x)
                # cacualte loss
                loss = criterion(model_output, label)
                # back propogation
                loss.backward()
                # weight update
                optimizer.step()

                # trn_loss summary
                trn_loss += loss.item()

                # #del (memory issue)
                # del loss
                # del model_output

            # validation
            with torch.no_grad():
                val_loss = 0.0
                for i, data in enumerate(val_loader):
                    x, val_label = data
                    if is_cuda:
                        x = x.cuda()
                        label = label.cuda()
                    val_output = self.m(x)
                    v_loss = criterion(val_output, val_label)
                    val_loss += v_loss

            # del v_loss
            # del val_output

            if epoch % 100 == 0:
                print("Epoch: {} / {} | train_loss: {:.5f} | val_loss: {:.4f}".format(epoch, num_epochs, trn_loss, val_loss))

            trn_loss_list.append(trn_loss)
            val_loss_list.append(val_loss)

        return trn_loss_list, val_loss_list

    def make_cluster(self):
        p_vectors = list()
        for user in self.user_lst:
            p_vectors.append(user.pref_vec)


        # for i in range(self.cluster_num[0], self.cluster_num[1]):
        #n_cluster = int(self.cluster_num[0])
        n_cluster = self.cluster_num

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

        for i in range(len(self.user_lst)):
            self.user_lst[i].set_cluster(kmeans.labels_[i])

        for cluster in self.cluster_lst:
            for i in range(len(kmeans.labels_)):
                if cluster.id == kmeans.labels_[i]:
                    cluster.add_user(self.user_lst[i])

            cluster.cal_p_k(self.contents_num)
        self.asso_server_to_cluster()

    def asso_server_to_cluster(self):
        idx_lst = list()
        while len(idx_lst) < len(self.server_lst):
            if len(idx_lst) < self.cluster_num:
                idx = random.randrange(self.cluster_num)
                if idx not in idx_lst:
                    idx_lst.append(idx)
            else:
                idx = random.randrange(self.cluster_num)
                idx_lst.append(idx)

        i = 0
        for s in self.server_lst:
            s.asso_cluster(self.cluster_lst[idx_lst[i]])
            i += 1


    def get_most_popular_contents(self):
        data_lst = [0 for _ in range(self.contents_num)]
        for u in self.user_lst:
            for i in range(len(data_lst)):
                data_lst[i] += u.pref_vec[i]

        idx_p_tuple = list()
        for i,v in enumerate(data_lst):
            idx_p_tuple.append((i, v))

        # sort
        idx_p_tuple.sort(key=lambda t: t[1], reverse=True)
        popularity_sorted = [e[0] for e in idx_p_tuple]
        return popularity_sorted


class Cluster:
    def __init__(self, id):
        self.id = id
        self.p_k = None
        self.cluster_users = list()

    def add_user(self, user):
        self.cluster_users.append(user)

    def cal_p_k(self, contents_num):
        tmp = [0 for _ in range(contents_num)]
        for u in self.cluster_users:
            for i in range(len(tmp)):
                tmp[i] += u.pref_vec[i]

        self.p_k = [element / len(self.cluster_users) for element in tmp]
        return self.p_k

    def get_popular_contents(self):
        idx_p_tuple = list()
        for i,v in enumerate(self.p_k):
            idx_p_tuple.append((i, v))

        # sort
        idx_p_tuple.sort(key=lambda t:t[1], reverse=True)
        popularity_sorted = [e[0] for e in idx_p_tuple]
        return popularity_sorted


class EdgeServer:
    def __init__(self, id, position):
        self.id = id
        self.position = position
        self.state = None
        self.content = None
        # self.total_request = 0
        self.algo_lst = list()
        self.cluster = None
        self.communication_r = None
        self.user_lst = list()

    def add_algo(self, algo):
        if type(algo).__name__ == 'CacheAlgo':
            self.algo_lst.append(algo)
            # print('Success to add algo')
        else:
            print('wrong algo class')

    def asso_cluster(self, cluster):
        self.cluster = cluster

    # def init_caching(self):
    #     for algo in self.algo_lst:
    #         if algo.is_cluster:
    #             data_lst = self.cluster.get_popular_contents()
    #         else:
    #             # data_lst = #zipf로 순서대로 넣어주기
    #             pass
    #         contents = algo.placement_content(data_lst)
    #         print("algo {}: {}".format(algo.id, contents))

    def request_content(self, content_id):
        # print('requests content (id: {})'.format(content_id))

        # self.total_request += 1
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
    def __init__(self, id, position):
        self.position = position
        self.pref_vec = list()
        self.cluster = None
        self.id = id
        self.capable_server_lst = list()
        self.state = False

    def make_pref(self, z_val, contents_num):
        self.pref_vec = [0 for _ in range(contents_num)]

        z = Zipf()
        z.set_env(z_val, contents_num)
        idx_lst = np.arange(0, contents_num)
        np.random.shuffle(idx_lst)
        tmp = copy.copy(z.pdf)

        for i in range(contents_num):
            self.pref_vec[idx_lst[i]] = tmp[i]

    def set_cluster(self, cluster_id):
        self.cluster = cluster_id

    def request(self, content_id):
        hit_lst = [0 for _ in range(len(self.capable_server_lst[0].algo_lst))]
        if len(self.capable_server_lst) > 0:
            for s in self.capable_server_lst:
                print("user {} requests the content {} at server {}".format(self.id, content_id, s.id))
                tmp = s.request_content(content_id)
                for i in range(len(hit_lst)):
                    if hit_lst[i] < 1:
                        hit_lst[i] += tmp[i]
        return hit_lst

class PPP:
    def __init__(self):
        #simulation window parameters
        self.lambda0 = None
        self.xx = None
        self.yy = None

    def set_env(self, density):
        r = 3
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

        position = list()
        for i in range(len(self.xx)):
            position.append((self.xx[i], self.yy[i]))

        return position


class Zipf:
    def __init__(self):
        self.pdf = None
        self.cdf = None

    def set_env(self, expn, num_contents):
        temp = np.power(np.arange(1, num_contents+1), -expn)
        zeta = np.r_[0.0, np.cumsum(temp)]
        self.pdf = [x / zeta[-1] for x in temp]
        self.cdf = [x / zeta[-1] for x in zeta]

    def get_sample(self):
        f = random.random()
        print(f)
        return np.searchsorted(self.cdf, f) - 1


if __name__ == '__main__':
    server = PPP()
    print(server.set_env(1))

    user = PPP()
    print(user.set_env(5))

    plt.scatter(server.xx, server.yy, edgecolors='b', facecolor='b', alpha=0.5)
    plt.scatter(user.xx, user.yy, edgecolors='r', facecolor='r', alpha=0.5)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    # user1 = User()
    # user1.make_pref(0.7, 10)
    # print(user1.pref_vec)
    #
    # user2 = User()
    # user2.make_pref(0.7, 10)
    # print(user2.pref_vec)