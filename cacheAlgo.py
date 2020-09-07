from elements import *

class CacheAlgo:
    def __init__(self):
        self.id = None
        self.is_cluster = False
        self.server_capacity = None
        self.LCU_num = None
        self.cached_contents_lst = list()
        self.users = list()
        self.cluster_num = None
        self.replacement_method = None
        self.cluster = None
        self.replacement_cnt = 0


    def set_option(self, id, is_cluster, cluster_num, capacity, LCU_num, rep_method='prediction'):
        self.id = id
        self.is_cluster = is_cluster
        self.cluster_num = cluster_num
        self.server_capacity = capacity
        self.LCU_num = LCU_num
        self.replacement_method = rep_method
        self.cached_contents_lst = [[] for _ in range(self.LCU_num)]

    def check_capacity(self):
        if self.cached_contents_lst < self.server_capacity:
            return True
        else:
            return False

    def asso_cluster(self, cluster_lst):
        self.cluster = [cluster_lst[i] for i in range(self.LCU_num)]


    def placement_content(self, contents_lst):
        if self.server_capacity > 0:
            for i in range(len(contents_lst)):
                for k in contents_lst[i]:
                    if len(self.cached_contents_lst[i]) >= self.server_capacity / self.LCU_num:
                        break
                    else:
                        if i != 0 and k in self.cached_contents_lst[i-1]:
                            continue
                        self.cached_contents_lst[i].append(k)

                # self.cached_contents_lst[i] = contents_lst[i][:int(self.server_capacity/self.LCU_num)]

        # if self.is_cluster:
        #     for i in range(self.LCU_num):
        #         self.cached_contents_lst[i].append(contents_lst[i][:self.server_capacity/len(contents_lst)])
        # else:
        #     self.cached_contents_lst.append(contents_lst[:self.server_capacity])

        # print("{}: {}".format(self.id, self.cached_contents_lst))

    def have_content(self, content_id):
        hit = False
        for i in range(len(self.cached_contents_lst)):
            if content_id in self.cached_contents_lst[i]:
                hit = True
                break
        return hit

    def replacement_content(self, t, new_content_id):
        evict_content = -1
        p_stack = list()
        if self.replacement_method == "None":
            pass
        elif self.replacement_method == "LRU":
            pass
        elif self.replacement_method == "LFU":
            pass
        elif self.replacement_method == "FIFO":
            pass
        elif self.replacement_method == "PREDICTION":
            for i in range(len(self.cluster)):
                t_p = self.cluster.predicted_p.loc[t]
                new_p = t_p[new_content_id]

                for cached_content in self.cached_contents_lst:
                    p_stack.append((cached_content, t_p[cached_content]))
                p_stack.sort(key=lambda t: t[1], reverse=True)
                least_id, least_p = p_stack[0]     #predicted popularity stack에서 least 인 것 하나 가져옴

                if least_p < new_p:
                    evict_content = least_id
                    self.cached_contents_lst.remove(least_id)   #캐시에서 popularity가 가장 낮은 content 삭제
                    self.cached_contents_lst.append(new_p)      #new content 삽입

        print("Evict: {} / New: {}".format(evict_content, new_content_id))
