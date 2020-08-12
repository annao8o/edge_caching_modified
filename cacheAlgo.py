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

    def set_option(self, id, is_cluster, cluster_num, capacity, LCU_num):
        self.id = id
        self.is_cluster = is_cluster
        self.cluster_num = cluster_num
        self.server_capacity = capacity
        self.LCU_num = LCU_num

    def have_content(self, content_id):
        hit = False
        for i in range(len(self.cached_contents_lst)):
            if content_id in self.cached_contents_lst[i]:
                hit = True
                break
        return hit

    def replacement_content(self, new_content_id):
        # 예측
        return