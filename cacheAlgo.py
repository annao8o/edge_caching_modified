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

    def set_option(self, id, is_cluster, cluster_num, capacity, LCU_num, rep_method):
        self.id = id
        self.is_cluster = is_cluster
        self.cluster_num = cluster_num
        self.server_capacity = capacity
        self.LCU_num = LCU_num
        self.replacement_method = rep_method

    def check_capacity(self):
        if self.cached_contents_lst < self.server_capacity:
            return True
        else:
            return False

    def placement_content(self, contents_lst):
        self.cached_contents_lst.append(contents_lst[:self.server_capacity])
        print("{}: {}".format(self.id, self.cached_contents_lst))

    def have_content(self, content_id):
        hit = False
        for i in range(len(self.cached_contents_lst)):
            if content_id in self.cached_contents_lst[i]:
                hit = True
                break
        return hit

    def replacement_content(self, new_content_id):
        if self.replacement_method == "None":
            pass
        elif self.replacement_method == "LRU":
            pass
        elif self.replacement_method == "LFU":
            pass
        elif self.replacement_method == "FIFO":
            pass
        elif self.replacement_method == "PREDICTION":
            pass