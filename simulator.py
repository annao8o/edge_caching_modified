import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from elements import *

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
    cluster_label = c.make_cluster()
    print("kmeans label:{}".format(cluster_label))

    current_time = datetime(2000, 1, 1, hour=6, minute=0, second=0)
    hit_result = [[] for _ in range(len(c.server_lst))]

    current_time += update_period
    while current_time <= end_time:
        pass




if __name__ == "__main__":
    c = Cloud()
    server_num = 3
    user_num = 200
    contents_num = 2000
    cluster_num = [0, 10]


    c.set_env(server_num, contents_num, cluster_num)

    z_val = 1.0
    arrival_rate = 10
    departure_rate = 1/60
    request_rate = 10

    simulation(c, z_val, contents_num, arrival_rate, departure_rate, request_rate)
