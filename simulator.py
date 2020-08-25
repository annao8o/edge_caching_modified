import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from elements import *
from cacheAlgo import *

np.random.seed(0)
def simulation(c, z_val, num_contents, arrival_rate, departure_rate, request_rate):
    update_period = timedelta(minutes=1)
    end_time = datetime(2000, 1, 1, hour=22, minute=0, second=0)

    zipf = Zipf()
    zipf.set_env(z_val, num_contents)

    ## generate and cluster users
    for u in c.user_lst:
        u.make_pref(z_val, num_contents)
    c.make_cluster()
    # print("kmeans label:{}".format(cluster_label))

    current_time = datetime(2000, 1, 1, hour=6, minute=0, second=0)
    # record_time = datetime(2000, 1, 1, hour=15, minute=0, second=0)
    hit_result = [0 for _ in range(len(c.server_lst[0].algo_lst))]

    # c.get_most_popular_contents()
    # for i in c.cluster_lst:
    #     print(i.id)
    #     i.get_popular_contents()


    for s in c.server_lst:
        for algo in s.algo_lst:
            data_lst = [[] for _ in range(algo.LCU_num)]
            if algo.is_cluster:
                print("\nserver: {} / algo: {} / asso_cluster: {}".format(s.id, algo.id, [algo.cluster[k].id for k in
                                                                                          range(len(algo.cluster))]))
                for i in range(algo.LCU_num):
                    data_lst[i] = algo.cluster[i].get_popular_contents()
            else:
                print("\nserver: {} / algo: {}".format(s.id, algo.id))
                data_lst[0] = c.get_most_popular_contents()
            algo.placement_content(data_lst)
        # s.init_caching()
    total_request = 0
    req_matrix = list()

    current_time += update_period
    daily_req = {i: 0 for i in range(num_contents)}

    while current_time <= end_time:

        # generate the users randomly
        arrive_user = np.random.poisson(arrival_rate)
        for _ in range(arrive_user):
            d_minute = np.random.exponential(1 / departure_rate)
            d_minute = round(d_minute)
            duration = timedelta(minutes=d_minute)
            c.arrive(current_time, duration)

        # 하루마다 daily_req 초기화
        if current_time.hour == 0 and current_time.minute == 0 and current_time.second == 0:
            c.req_mat.append(daily_req)
            daily_req = {i: 0 for i in range(num_contents)}

        print(current_time)

        # generate the requests randomly (각 서버의 p_k에 따라 request 생성하는 것으로 수정 필요)
        requests = np.random.poisson(request_rate)

        req_user = list()
        for _ in range(requests):
            random_u = c.user_lst[random.randrange(len(c.user_lst))]
            if random_u.state and random_u not in req_user:
                req_user.append(random_u)

        for u in req_user:
            total_request += 1
            zipf_random = zipf.get_sample()
            requested_content, hit_lst = u.request(zipf_random)
            print(hit_lst)
            for i in range(len(hit_lst)):
                hit_result[i] += hit_lst[i]

            print(hit_result, '\n')

            daily_req[requested_content] += 1

        # update the current users based on their departure time
        c.update(current_time)
        current_time += update_period

    result = {'total_request': total_request, 'hit_count': hit_result, 'hit_ratio': np.array(hit_result) / total_request}
    print(result)


if __name__ == "__main__":
    c = Cloud()

    server = PPP()
    server_position = server.set_env(0.5)
    user = PPP()
    user_position = user.set_env(4)

    # server_num = 5
    # user_num = 200
    contents_num = 2000
    cluster_num = 3
    server_communication_r = 1.0
    z_val = 1.0
    arrival_rate = 1
    departure_rate = 1/60
    request_rate = 10

    #num_batches = len(train_loader)
    num_epochs = 2000
    learning_rate = 0.001
    input_size = 1
    hidden_size = 100
    num_layers = 1
    output_size = 1

    model = LSTM(input_size, output_size, hidden_size, num_layers)
    c.set_env(server_position, user_position, contents_num, cluster_num, model)

    #c.training(learning_rate, num_epochs, train_loader, val_loader)

    for s in c.server_lst:
        s.communication_r = server_communication_r

        ## find the communicatable points
        for u in c.user_lst:
            if (s.position[0] - u.position[0])**2 + (s.position[1] - u.position[1])**2 <= s.communication_r**2:
                s.user_lst.append(u)
                u.capable_server_lst.append(s)
                u.state = True

        algo_0 = CacheAlgo()
        algo_0.set_option('algo_0', False, 1, 0, 1)
        algo_1 = CacheAlgo()
        algo_1.set_option('algo_1', False, 1, 200, 1)
        algo_2 = CacheAlgo()
        algo_2.set_option('algo_2', False, 1, 400, 1)
        algo_3 = CacheAlgo()
        algo_3.set_option('algo_3', False, 1, 600, 1)
        algo_4 = CacheAlgo()
        algo_4.set_option('algo_4', False, 1, 800, 1)
        algo_5 = CacheAlgo()
        algo_5.set_option('algo_5', False, 1, 1000, 1)
        algo_6 = CacheAlgo()
        algo_6.set_option('algo_6', False, 1, 1200, 1)
        algo_7 = CacheAlgo()
        algo_7.set_option('algo_7', False, 1, 1400, 1)
        algo_8 = CacheAlgo()
        algo_8.set_option('algo_8', False, 1, 1600, 1)
        algo_9 = CacheAlgo()
        algo_9.set_option('algo_9', False, 1, 1800, 1)
        algo_10 = CacheAlgo()
        algo_10.set_option('algo_10', False, 1, 2000, 1)

        algo_11 = CacheAlgo()
        algo_11.set_option('algo_11', True, cluster_num, 0, 1)
        algo_12 = CacheAlgo()
        algo_12.set_option('algo_12', True, cluster_num, 200, 1)
        algo_13 = CacheAlgo()
        algo_13.set_option('algo_13', True, cluster_num, 400, 1)
        algo_14 = CacheAlgo()
        algo_14.set_option('algo_14', True, cluster_num, 600, 1)
        algo_15 = CacheAlgo()
        algo_15.set_option('algo_15', True, cluster_num, 800, 1)
        algo_16 = CacheAlgo()
        algo_16.set_option('algo_16', True, cluster_num, 1000, 1)
        algo_17 = CacheAlgo()
        algo_17.set_option('algo_17', True, cluster_num, 1200, 1)
        algo_18 = CacheAlgo()
        algo_18.set_option('algo_18', True, cluster_num, 1400, 1)
        algo_19 = CacheAlgo()
        algo_19.set_option('algo_19', True, cluster_num, 1600, 1)
        algo_20 = CacheAlgo()
        algo_20.set_option('algo_20', True, cluster_num, 1800, 1)
        algo_21 = CacheAlgo()
        algo_21.set_option('algo_21', True, cluster_num, 2000, 1)

        algo_22 = CacheAlgo()
        algo_22.set_option('algo_22', True, cluster_num, 0, 2)
        algo_23 = CacheAlgo()
        algo_23.set_option('algo_23', True, cluster_num, 200, 2)
        algo_24 = CacheAlgo()
        algo_24.set_option('algo_24', True, cluster_num, 400, 2)
        algo_25 = CacheAlgo()
        algo_25.set_option('algo_25', True, cluster_num, 600, 2)
        algo_26 = CacheAlgo()
        algo_26.set_option('algo_26', True, cluster_num, 800, 2)
        algo_27 = CacheAlgo()
        algo_27.set_option('algo_27', True, cluster_num, 1000, 2)
        algo_28 = CacheAlgo()
        algo_28.set_option('algo_28', True, cluster_num, 1200, 2)
        algo_29 = CacheAlgo()
        algo_29.set_option('algo_29', True, cluster_num, 1400, 2)
        algo_30 = CacheAlgo()
        algo_30.set_option('algo_30', True, cluster_num, 1600, 2)
        algo_31 = CacheAlgo()
        algo_31.set_option('algo_31', True, cluster_num, 1800, 2)
        algo_32 = CacheAlgo()
        algo_32.set_option('algo_32', True, cluster_num, 2000, 2)

        # s.add_algo(algo_0)
        # s.add_algo(algo_1)
        s.add_algo(algo_2)
        # s.add_algo(algo_3)
        # s.add_algo(algo_4)
        # s.add_algo(algo_5)
        # s.add_algo(algo_6)
        # s.add_algo(algo_7)
        # s.add_algo(algo_8)
        # s.add_algo(algo_9)
        # s.add_algo(algo_10)
        #
        # s.add_algo(algo_11)
        # s.add_algo(algo_12)
        # s.add_algo(algo_13)
        s.add_algo(algo_14)
        # s.add_algo(algo_15)
        # s.add_algo(algo_16)
        # s.add_algo(algo_17)
        # s.add_algo(algo_18)
        # s.add_algo(algo_19)
        # s.add_algo(algo_20)
        # s.add_algo(algo_21)

        # s.add_algo(algo_22)
        # s.add_algo(algo_23)
        # s.add_algo(algo_24)
        s.add_algo(algo_25)
        # s.add_algo(algo_26)
        # s.add_algo(algo_27)
        # s.add_algo(algo_28)
        # s.add_algo(algo_29)
        # s.add_algo(algo_30)
        # s.add_algo(algo_31)
        # s.add_algo(algo_32)


    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(server.xx, server.yy, edgecolors='r', facecolor='r', alpha=0.5)
    ax.scatter(user.xx, user.yy, edgecolors='b', facecolor='b', alpha=0.5)
    for s in c.server_lst:
        ax.annotate(s.id, xy=s.position)
        circle = plt.Circle(s.position, s.communication_r, edgecolor='r', facecolor='None', alpha=0.5)
        ax.add_artist(circle)
        # print(s.id, s.user_lst)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    simulation(c, z_val, contents_num, arrival_rate, departure_rate, request_rate)
