#!/PycharmProjects/env python
#-*- coding:utf-8 -*-
import pickle
import numpy as np
import datetime
from util import *
import torch.nn as nn

class DataHandler:
    def __init__(self,cache_dir,SESSION_LEN,WINDOWS_size):
        self.SESSION_LEN =SESSION_LEN
        self.WINDOWS_size =WINDOWS_size
        self.cache_dir=cache_dir
        self.DATASET_path = cache_dir+str(self.SESSION_LEN)+'i/5train_val_test'+str(self.WINDOWS_size)+'_W.pickle'
        #self.id_to_name=pickle.load(open(cache_dir + str(self.SESSION_LEN) +'i/2map_to_name.pickle', 'rb'))
        DATAS = pickle.load(open(self.DATASET_path, 'rb'))
        self.interactions = pickle.load(open(self.cache_dir + str(self.SESSION_LEN) + 'i/2interacted.pickle', 'rb'))
        self.trainset = DATAS['trainset']
        self.validset = DATAS['validset']
        self.testset = DATAS['testset']
        self.global_timedlt = DATAS['global_timedlt']
        item_map, user_map, _ = pickle.load(open(self.cache_dir + str(self.SESSION_LEN) + 'i/2item_user_time_mapped.pickle', 'rb'))
        self.num_items = len(item_map) + 1  # 补全时多了一个0
        self.num_users = len(user_map)
        self.items_set =set(range(1, self.num_items))
        #self.statistic=DATAS['users-train_valid_test_samples']

    def get_neg(self,batch_users,batch_gts, max_gt_len):
        batch_neg = []
        batch_gt = []
        for i in range(len(batch_users)):
           uid=batch_users[i]
           gt=batch_gts[i]
           if len(gt) < max_gt_len:
               gt = np.resize(gt, max_gt_len)
           else:
               gt = np.array(gt, dtype=np.int32)
           batch_gt.append(gt)

           positive = self.interactions[uid]
           neg_items_list = list(self.items_set - set(positive))  # 差集
           negs = np.random.choice(neg_items_list, max_gt_len)  # 从neg_items_list中随机采样负样本 len(batch_gt_items[index])
           negs_list = np.ndarray.tolist(negs)
           batch_neg.append(negs_list)
        batch_gt = np.array(batch_gt, dtype=np.int32)
        batch_neg =np.array(batch_neg,dtype=np.int32)
        return batch_gt,batch_neg

    def log_config(self,message):
        timestamp = str(datetime.datetime.now())
        config ='\n'+ timestamp + '\n' + message
        with open(self.cache_dir+str(self.SESSION_LEN)+'i/log.txt', 'a+') as f:#parameterlog
            f.write(config)

    def log_attention(self, message):
        timestamp = str(datetime.datetime.now())
        config = '\n'+timestamp + '\n' + message
        with open(self.cache_dir + str(self.SESSION_LEN) + 'i/inter_attention.txt', 'a+') as f:
            f.write(config)



