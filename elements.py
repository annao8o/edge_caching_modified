import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import random
import networkx as nx
import copy
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from prediction_model import *
from math import hypot
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
        self.moved_users = list()   #user mobility (arrive/depart) 관리하기 위한 list
        self.cluster_num = None
        self.m = None
        self.req_mat = list()
        self.total_arrive = None
        self.total_depart = None
        self.cluster_centers = None

    def set_env(self, server_position, user_position, contents_num, cluster_num, model):
        self.total_arrive = 0
        self.total_depart = 0
        self.server_position = server_position
        self.user_position = user_position
        self.contents_num = contents_num
        self.cluster_num = cluster_num
        self.m = model
        self.req_dict = {i:0 for i in range(self.contents_num)}
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

    def record_request(self, daily_req):
        self.req_mat.append(daily_req)

    def arrive(self, time, duration):
        # user 위치 랜덤으로 지정
        p = random.random() * 2 * math.pi
        r = 3 * math.sqrt(random.random())
        x = math.cos(p) * r
        y = math.sin(p) * r
        position = (x, y)
        u = User(len(self.user_lst) + 1, position)
        u.make_pref(1.0, self.contents_num)
        # self.assign_cluster(u)

        for s in self.server_lst:
            if (s.position[0] - u.position[0])**2 + (s.position[1] - u.position[1])**2 <= s.communication_r**2:
                s.user_lst.append(u)
                u.capable_server_lst.append(s)
                u.state = True

        self.user_lst.append(u)

        expirate_time = time + duration
        new_user = (expirate_time, u)

        self.moved_users.append(new_user)
        self.moved_users = sorted(self.moved_users, key=lambda users: users[0])

        self.total_arrive += 1
        print('User {} arrives at {} (from {} to {})'.format(u.id, u.position, time, expirate_time))

        # for algo in self.algo_lst:
        #     algo.arrive_user(new_user[1])  # new_user 객체를 넘겨줌

    def depart(self):
        if self.moved_users:
            d_u = self.moved_users.pop(0)
            self.total_depart += 1
            self.user_lst.remove(d_u[1])
            if d_u[1].state:
                for s in d_u[1].capable_server_lst:
                    if d_u[1] in s.user_lst:
                        s.user_lst.remove(d_u[1])
            print("user {} departs.".format(d_u[1].id))
            return d_u[1]

    def update(self, time):
        depart_user_lst = list()
        while self.moved_users:
            if self.moved_users[0][0] <= time:
                depart_user_lst.append(self.depart())
            else:
                break

        # return depart_user_lst

    def training(self, learning_rate, num_epochs, data, current_t, window_size):
        data = self.req_mat[current_t-1-window_size:current_t-1]

        sc = MinMaxScaler()
        train_data = sc.fit_transform(data)

        seq_length = 7
        x, y = create_sequences(train_data, seq_length)

        trainX = Variable(torch.Tensor(x))
        trainY = Variable(torch.Tensor(y))

        batch_size = 100

        train_loader = DataLoader(train_data, shuffle=False, batch_size=batch_size)
        # val_loader =

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.m.parameter(), lr=learning_rate)

        trn_loss_list = model_training(self.m, learning_rate, num_epochs, train_loader)

    def prediction(self):
        self.m.eval()
        test_predict = self.m

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
        self.cluster_centers = kmeans.cluster_centers_

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

    # def assign_cluster(self, user):
    #     if self.cluster_num > 1:
    #         tmp = []
    #         data = user.pref_vec
    #
    #         for i in range(len(self.cluster_centers)):
    #             dist = hypot(data[0] - self.cluster_centers[i][0], data[1] - self.cluster_centers[i][1])
    #             tmp.append(dist)
    #         nearest = tmp[0]
    #         idx = 0
    #         for i in range(len(tmp)):
    #             if tmp[i] < nearest:
    #                 nearest = tmp[i]
    #                 idx = i
    #     else:
    #         idx = 0
    #
    #     self.label_lst[idx].append(new_user.userId)


    def asso_server_to_cluster(self):
        # idx_lst = list()
        # while len(idx_lst) < len(self.server_lst):
        #     if len(idx_lst) < self.cluster_num:
        #         idx = random.randrange(self.cluster_num)
        #         if idx not in idx_lst:
        #             idx_lst.append(idx)
        #     else:
        #         idx = random.randrange(self.cluster_num)
        #         idx_lst.append(idx)
        for s in self.server_lst:
            random.shuffle(self.cluster_lst)
            for algo in s.algo_lst:
                if algo.is_cluster:
                    algo.asso_cluster(self.cluster_lst)


    def get_most_popular_contents(self):
        data_lst = [0 for _ in range(self.contents_num)]
        for u in self.user_lst:
            for i in range(len(data_lst)):
                data_lst[i] += u.pref_vec[i]

        idx_p_tuple = list()
        tmp = list()
        for i,v in enumerate(data_lst):
            idx_p_tuple.append((i, v))
            tmp.append(v)

        print(tmp)

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
        tmp = list()
        for i,v in enumerate(self.p_k):
            idx_p_tuple.append((i, v))
            tmp.append(v)

        print(tmp)

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

    # def asso_cluster(self, cluster):
    #     for algo in self.algo_lst:
    #         algo.cluster =
    #     self.cluster = cluster

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
                # algo.replacement_content(content_id)

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

    def request(self, zipf_random):
        idx_p_tuple = list()
        for i, v in enumerate(self.pref_vec):
            idx_p_tuple.append((i, v))  #i: index, v: popularity

        # sort
        idx_p_tuple.sort(key=lambda t: t[1], reverse=True)
        popularity_sorted = [e[0] for e in idx_p_tuple]

        hit_lst = [0 for _ in range(len(self.capable_server_lst[0].algo_lst))]
        if len(self.capable_server_lst) > 0:
            for s in self.capable_server_lst:
                print("user {} requests the content {} at server {}".format(self.id, popularity_sorted[zipf_random], s.id))
                tmp = s.request_content(popularity_sorted[zipf_random])
                for i in range(len(hit_lst)):
                    if hit_lst[i] < 1:
                        hit_lst[i] += tmp[i]
        return popularity_sorted[zipf_random], hit_lst

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