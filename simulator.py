import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from elements import *
from cacheAlgo import *

np.random.seed(0)
def simulation(c, z_val, contents, arrival_rate, departure_rate, request_rate):
    update_period = timedelta(minutes=1)
    end_time = datetime(2000, 1, 1, hour=22, minute=0, second=0)

    zipf = Zipf()
    zipf.set_env(z_val, contents)

    ## generate and cluster users
    for i in range(user_num):
        u = User(i)
        u.make_pref(z_val, contents)
        c.add_user(u)
    c.make_cluster()
    # print("kmeans label:{}".format(cluster_label))

    current_time = datetime(2000, 1, 1, hour=6, minute=0, second=0)
    hit_result = [[] for _ in range(len(c.server_lst))]

    for s in c.server_lst:
        print("\nserver: {} / asso_cluster: {}".format(s.id, s.cluster.id))
        s.init_caching()

    current_time += update_period
    while current_time <= end_time:
        for s in c.server_lst:
            requests = np.random.poisson(request_rate)
            for _ in range(requests):
                content = zipf.get_sample()
                hit_lst = s.request_content(content)
                for i in range(len(s.algo_lst)):
                    hit_result[i] += hit_lst[i]






if __name__ == "__main__":
    c = Cloud()
    server_num = 5
    user_num = 200
    contents_num = 10
    cluster_num = 3


    c.set_env(server_num, contents_num, cluster_num)

    z_val = 1.0
    arrival_rate = 1
    departure_rate = 1/60
    request_rate = 10

    for s in c.server_lst:
        algo_0 = CacheAlgo()
        algo_0.set_option('algo_0', True, cluster_num, 10, 1)
        algo_1 = CacheAlgo()
        algo_1.set_option('algo_1', True, cluster_num, 20, 1)

        s.add_algo(algo_0)
        s.add_algo(algo_1)

    simulation(c, z_val, contents_num, arrival_rate, departure_rate, request_rate)
