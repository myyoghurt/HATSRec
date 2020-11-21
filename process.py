 #!/PycharmProjects/env python
#-*- coding:utf-8 -*-
import pandas as pd
import dateutil.parser
import pickle
import numpy as np
import time
import json
import math
import matplotlib.pyplot as plt
from copy import deepcopy
from util import *
from itertools import groupby
import datetime
from collections import Counter

runtime = time.time()
DATASET_DIR="E:/CODE/DATASETS"
Amanzon="/Amazon_Video"
Movie="/MovieLens1m"
Lastfm="/Lastfm"
Amazon_DATASET_FILE=DATASET_DIR+Amanzon+'/ratings_Video_Games.csv' #ratings_Books ratings_Video_Games
Movie_FILE=DATASET_DIR+Movie+'/ratings.dat'
Lastfm_FILE=DATASET_DIR+Lastfm+'/userid-timestamp-artid-artname-traid-traname.tsv'
DATASET=Lastfm
cache_dir=DATASET_DIR+DATASET+'/'
SESSIONS_Timedlt =2
Max_SESSION_LEN =30
Min_SESSION_LEN=2
Min_SESSIONS=3
WINDOWS_size=12
PAD_VALUE = 0
MIN_USER_counts=10#10
MIN_Item_counts=10#10
global_timedlt={}

def Data_read():
    if "Lastfm" in DATASET:
        dataset_list = []
        with open(Lastfm_FILE, 'rt', buffering=10000, encoding='utf8') as dataset:
            for line in dataset:
                line = line.split('\t')
                user_id = line[0]
                timestamp = (dateutil.parser.parse(line[1])).timestamp()
                artist_id = line[2]
                dataset_list.append((user_id, artist_id, None, timestamp))
        ratings_data = pd.DataFrame(dataset_list, columns=['user_id', 'item_id', 'rating', 'timestamp'],
                                    dtype='float32')
    else:
        r_columns = ['user_id', 'item_id', 'rating', 'timestamp']
        if "Amazon" in DATASET:
            ratings_data = pd.read_csv(Amazon_DATASET_FILE, sep=',', names=r_columns, engine='python', iterator=True)
        if "MovieLens" in DATASET:
            ratings_data = pd.read_csv(Movie_FILE, sep='::', names=r_columns, engine='python', iterator=True)

        loop = True
        chunkSize = 10000
        chunks = []
        index = 0
        while loop:
            try:
                chunk = ratings_data.get_chunk(chunkSize)
                chunks.append(chunk)
                index += 1
            except StopIteration:
                loop = False
                print('StopIteration')
        print('combination')
        ratings_data = pd.concat(chunks, ignore_index=True)
    Statistics_Before(ratings_data,'ratings_data')
    return ratings_data

def Statistics_Before(data,state):
    df=deepcopy(data)
    users = list(df['user_id'].unique())  # 列出该列的唯一值
    users_num = len(users)  # 统计该列有多少个不一样的值
    items = list(df['item_id'].unique())
    items_num = len(items)
    total_interactions = len(df)
    df_ordered = df.sort_values(['user_id'], ascending=True)
    user_items= {}
    user_times={}
    timedlt_list=[]
    for interaction in df_ordered.iterrows():
        user_id=interaction[1][0]
        item_id = interaction[1][1]
        stamp = interaction[1][3]
        stamp = time.strftime("%Y%m%d%H%M", time.localtime(stamp))  # 转为时间格式1995-10-01 05
        year = int(stamp[0:4])
        month = int(stamp[4:6])
        day = int(stamp[6:8])
        hour = int(stamp[8:10])
        timestamp = (year * 365 + month * 30 + day) * 24 + hour

        '''若是新用户，则创建新set'''
        if user_id not in user_items.keys():
            user_items[user_id] = []
            user_times[user_id]=[timestamp]
            user_items[user_id].append(item_id)#记录每个用户交互的unique item
            continue

        user_items[user_id].append(item_id)
        last_timestamp = user_times[user_id][-1]
        item_dlt = abs(timestamp - last_timestamp)
        timedlt_list.append(item_dlt)#记录所有用户items之间交互时间差
        user_times[user_id].append(timestamp)
    save_pickle(timedlt_list, cache_dir+'timedlt')

    data = load_pickle(cache_dir + 'timedlt')
    timedlt_list = deepcopy(data)
    # dic = {}
    # t = sorted(timedlt_list)
    # for k, g in groupby(t,key=lambda x: x // 24):
    #     dic['{}-{}'.format(k * 24, (k + 1) * 24 - 1)] = len(list(g))
    # print("dic\n", dic)
    # data_save_to_json(dic, cache_dir+'timedlt')

    num_interaction_list = []
    for key, items in user_items.items():
        num_interaction_list.append(len(items))
    '''每个用户交互数分布'''
    num_interaction_dic = {}
    num_interaction_t = sorted(num_interaction_list)
    for k, g in groupby(num_interaction_t, key=lambda x: x // 10):  # 10个item 为一个区间
        num_interaction_dic['{}-{}'.format(k * 10, (k + 1) * 10 - 1)] = len(list(g))
    interaction_distribute = sorted(num_interaction_dic.items(), key=lambda x: x[1],
                                    reverse=True)  # 对value降序,
    '''所有交互时间间隔分布'''
    timedelta_dic = {}
    timedelta_t = sorted(timedlt_list)
    for k, g in groupby(timedelta_t, key=lambda x: x // 10):  # 10hours 为一个区间
        timedelta_dic['{}-{}'.format(k * 10, (k + 1) * 10 - 1)] = len(list(g))
    timedelta_distribute = sorted(timedelta_dic.items(), key=lambda x: x[1], reverse=True)  # 对value降序

    avg_actions_user=np.mean(num_interaction_list)
    Density=total_interactions/float(users_num*items_num)

    timestamp = str(datetime.datetime.now())
    config = '\n' + timestamp + '\nBEFORE:remove_unfrequent_users and items:[' + str(
        MIN_USER_counts) + ',' + str(MIN_Item_counts) + '] ||state:' + str(state) + \
             '\n have users_num:' + str(users_num) + '|| items_num:' + str(items_num) + '|| total_interactions:' + str(
        int(total_interactions)) + \
             '\navg_actions_user:' + str(round(avg_actions_user, 3)) + '|| Density:' + str(round(Density, 8))
    print(config)
    with open(cache_dir+'STATISTIC.txt', 'a+') as f:
        f.write(config + '\n')
    distribute_config = '\n' + timestamp + 'state:' + str(state) + '\ninteraction_distribute||total user:' + str(
        users_num) + '\t' + str(interaction_distribute) + \
                        '\ntimedelta_distribute|| num_timedlts:' + str(len(timedlt_list)) + '\t' + str(
        timedelta_distribute)
    with open(cache_dir+'distribute_STATISTIC.txt', 'a+') as f:
        f.write(distribute_config + '\n')
''' 1 移除交互数据少于min_counts条的user'''
def remove_unfrequent(data):
    df1 = deepcopy(data)
    item_counts = df1['item_id'].value_counts()
    df1 = df1[df1["item_id"].isin(item_counts[item_counts >= MIN_Item_counts].index)]
    user_counts = df1['user_id'].value_counts()
    df1 = df1[df1["user_id"].isin(user_counts[user_counts >= MIN_USER_counts].index)]
    
    Statistics_Before(df1, 'remove_unfrequent_users and item')
    save_pickle(df1, cache_dir+'1removed')

def Statistics_After(USER_SESSIONS,items_num,state):
    NUM_SESSION_list=[]
    TOTAL_USER=len(USER_SESSIONS)
    TOTAL_MAX_SE_LEN = []
    session_len_list=[]
    num_interaction_list=[]
    than_max_len = 0
    for uid, sessions in USER_SESSIONS.items():  # 总sessions; users;  items; interactions; n_sessionsions; avg sessions per user; avg length per session
        NUM_SESSION_list.append(len(sessions))
        lengths = [len(session) for session in sessions]
        session_len_list.extend(lengths)
        than_max_len += (np.array(lengths) >= Max_SESSION_LEN).sum()
        num_interaction_list.append(np.sum(lengths))
        TOTAL_MAX_SE_LEN.append(np.max(lengths))
    than_window = (np.array(NUM_SESSION_list) >=WINDOWS_size).sum()
    TOTAL_interactions = np.sum(session_len_list)
    '''session个数'''
    n_sessionSION = np.sum(NUM_SESSION_list)
    MIN_SESSIONS = np.min(NUM_SESSION_list)
    MAX_SESSIONS = np.max(NUM_SESSION_list)
    than_window_percent = than_window / float(TOTAL_USER)
    avg_sessions_per_user = n_sessionSION / float(TOTAL_USER)
    session_num_dic = {}
    session_num_t = sorted(NUM_SESSION_list)
    for k, g in groupby(session_num_t, key=lambda x: x // 5):  # 5个session 为一个区间
        session_num_dic['{}-{}'.format(k * 5, (k + 1) * 5 - 1)] = len(list(g))
    session_num_distribute = sorted(session_num_dic.items(), key=lambda x: x[1],
                                    reverse=True)  # 对value降序, 每个用户session个数分布
    '''session length'''
    max_len_a_session = np.max(TOTAL_MAX_SE_LEN)
    avg_length_per_session = TOTAL_interactions / float(n_sessionSION)
    than_max_len_percent = than_max_len / float(n_sessionSION)
    session_len_dic = {}
    session_len_t = sorted(session_len_list)
    for k, g in groupby(session_len_t, key=lambda x: x // 5):  # 5个session 为一个区间
        session_len_dic['{}-{}'.format(k * 5, (k + 1) * 5 - 1)] = len(list(g))
    session_len_distribute = sorted(session_len_dic.items(), key=lambda x: x[1],
                                    reverse=True)  # 对value降序, session长度分布
    '''每个用户交互数分布'''
    avg_action_per_user = TOTAL_interactions / float(TOTAL_USER)
    num_interaction_dic = {}
    num_interaction_t = sorted(num_interaction_list)
    for k, g in groupby(num_interaction_t, key=lambda x: x // 10):  # 10个item 为一个区间
        num_interaction_dic['{}-{}'.format(k * 10, (k + 1) * 10 - 1)] = len(list(g))
    interaction_distribute = sorted(num_interaction_dic.items(), key=lambda x: x[1],
                                    reverse=True)  # 对value降序,
    Density = TOTAL_interactions / float(TOTAL_USER * items_num)
    timestamp = str(datetime.datetime.now())
    config = '\n' + timestamp + '\nAFTER:Max_SESSION_LEN,Min_SESSION_LEN,WINDOWS_size,SESSIONS_Timedlt:[' + str(Max_SESSION_LEN) + '_i,'+str(Min_SESSION_LEN)+'_i,' + str(
        int(WINDOWS_size)) +'_W,'+str(int(SESSIONS_Timedlt))+ 'hour] ||state:' + str(state) + \
             '\nusers_num:' + str(TOTAL_USER) + '||items_num:'+str(items_num)+ '||total interactions:' + str(TOTAL_interactions) + '||avg_action_per_user:'+str(avg_action_per_user)+\
             '||n_sessionSION:' + str(n_sessionSION) +\
             '\navg_sessions_per_user:' + str(round(avg_sessions_per_user, 3)) + '||avg_length_per_session:' + str(
        round(avg_length_per_session, 3)) + \
             '\nmaxlen_a_session:' + str(max_len_a_session) + '||than_max_len_percent:' + str(
        round(than_max_len_percent, 3)) + \
             '\nMIN_SESSIONS:' + str(MIN_SESSIONS) + '||MAX_SESSIONS:' + str(
        MAX_SESSIONS) + '||than_window_percent:' + str(round(than_window_percent, 3))+"||Density:"+str(round(Density,5))
    print(config)
    with open(cache_dir + 'STATISTIC.txt', 'a+') as f:
        f.write(config + '\n')
        
    # distribute_config = '\n' + timestamp + '\nAFTER:Max_SESSION_LEN,Min_SESSION_LEN,WINDOWS_size,SESSIONS_Timedlt:[' + str(Max_SESSION_LEN) + '_i,'+str(Min_SESSION_LEN)+'_i,' + str(
    #     int(WINDOWS_size)) +'_W,'+str(int(SESSIONS_Timedlt))+ 'hour] ||state:' + str(state) + \
    #          '\nusers_num:' + str(TOTAL_USER)  + \
    #          '\ninteraction_distribute||total TOTAL_USER:' + str(TOTAL_USER) + '||avg_action_per_user:' + str(
    #     avg_action_per_user) + '\t' + str(interaction_distribute) + \
    #          '\nn_sessionSION:' + str(n_sessionSION) + \
    #          '\nsession_num_distribute||avg_sessions_per_user:' + str(
    #     round(avg_sessions_per_user, 3)) + '|| MIN_SESSIONS:' + str(MIN_SESSIONS) + '||MAX_SESSIONS:' + str(
    #     MAX_SESSIONS) + \
    #          '\n divide TOTAL_USER:' + str(TOTAL_USER) + '\t' + str(session_num_distribute) + \
    #          '\nsession_len_distribute||avg_length_per_session:' + str(
    #     round(avg_length_per_session, 3)) + '||maxlen_a_session:' + str(max_len_a_session) + \
    #          '\n divide n_sessionSION:' + str(n_sessionSION) + '\t' + str(session_len_distribute)
    # print(distribute_config)
    # with open(cache_dir + 'distribute_STATISTIC.txt', 'a+') as f:
    #     f.write(distribute_config + '\n')

'''2 为每个user人工划分session序列'''
def split_sessions():
    df1 = load_pickle(cache_dir+'1removed')    
    df_ordered = df1.sort_values(['user_id', 'timestamp'], ascending=True)
    items = list(df1['item_id'].unique())
    items_num = len(items)
    user_sessions = {}
    for interaction in df_ordered.iterrows():  # ['user_id':111,'item_id':222, 'rating':3, 'timestamp':201811011010]
        user_id =interaction[1][0]
        item_id = interaction[1][1]
        timestamp=interaction[1][3]
        time1 = time.strftime("%Y%m%d%H%M", time.localtime(timestamp))  # 转为时间格式1995-10-01 05:00
        year,month,day,hour,minute = int(time1[0:4]),int(time1[4:6]),int(time1[6:8]),int(time1[8:10]),int(time1[10:11])
        time2 = (year*365 + month*30 + day)*24+hour
        new_interaction = [item_id, time2]

        '''若是新用户，则创建新sessions'''
        if user_id not in user_sessions.keys():
            user_sessions[user_id] = []
            current_session = [new_interaction]
            user_sessions[user_id].append(current_session)
            continue
        '''
        若是已存在的user，则比较当前interaction与上一次interaction的时间差，
        若小于timedelta，则当前interaction加入当前session
        若大于timedelta，则创建新session,且当前interaction加入新session
        '''
        last_interaction=current_session[-1]
        last_time = last_interaction[-1]
        if new_interaction!=last_interaction:
            timedelta = abs(time2 - last_time)
            if timedelta <= SESSIONS_Timedlt: # Movie 2 hours,Amazon 2 day
                current_session.append(new_interaction)
            else:
                current_session = [new_interaction]
                user_sessions[user_id].append(current_session)

    Statistics_After(user_sessions, items_num,'2user_session')

    # 划分过长的session和移除session长度小于2的session
    new_user_sessions,global_min_timestamp = split_remove_sessions(user_sessions)
    print("...mapping...")
    user_map,item_map,time_map= map_ids_to_labels(new_user_sessions,global_min_timestamp)
    n_session=0
    mapped_user_sessions={}
    min_scale_timedlts={}
    max_scale_timedlts={}
   
    for key, val in new_user_sessions.items():
        uid=user_map[key]
        n_session+=len(val)
        session_timedlt_list=[]
        sessions_list=[]
        t_end = 0
        for session in val:
            event_list=list(map(lambda x: [item_map[x[0]],time_map[x[1]]], session))
            if len(session_timedlt_list)<1:
                t_end=event_list[0][-1]
                continue
            t_start = event_list[0][-1]
            dlt=t_start-t_end
            session_timedlt_list.append(dlt)
            t_end=event_list[-1][1]
            sessions_list.append(event_list)

        mapped_user_sessions[uid] = sessions_list
        min_scale_timedlts[uid],max_scale_timedlts[uid]=min(session_timedlt_list),max(session_timedlt_list)#取session间时间间隔min作绝对范围/max作相对范围

    Statistics_After(mapped_user_sessions,len(item_map),'mapped_user_sessions')
    pickle_dict = {}
    pickle_dict['user_sessions'] = mapped_user_sessions
    pickle_dict['total_sessions'] = n_session
    pickle_dict['scale_timedlts']=[min_scale_timedlts,max_scale_timedlts]
    save_pickle(pickle_dict, cache_dir+str(Max_SESSION_LEN)+'i/3user_sessions')

'''过长的session，如长度大于Max_SESSION_LENGTH=10的session划分为多个session'''
def split_remove_sessions(user_sessions):
    new_user_sessions = {}
    global_min_timestamp=float('inf')
    for key, val in user_sessions.items():
        splitted_sessions = []
        for session in val:
            splitted_sessions += split_single_session(session)
            global_min_timestamp=min(global_min_timestamp,np.min([event[1] for event in session]))
        if len(splitted_sessions)>=Min_SESSIONS:
            new_user_sessions[key]=splitted_sessions
    return new_user_sessions,global_min_timestamp

def split_single_session(session):
    splitted = [session[i:i+Max_SESSION_LEN] for i in range(0, len(session), Max_SESSION_LEN)]
    if len(splitted[-1]) < 2:  #移除session长度小于Min_SESSION_LEN的session
        del splitted[-1]
    return splitted

''' 3 将数据集中的unique userid和unique itemid标签化'''
def map_ids_to_labels(filter_session_data,global_min_timestamp):
    user_sessions =deepcopy(filter_session_data)
    user_map = {}
    item_map = {}
    time_map={}
    interacted={}
    unique_interactions_num=0
    for key, session_list in user_sessions.items():
        interacted_set = set()
        if key not in user_map:
            uid = len(user_map)
            user_map[key] = uid
        for session in session_list:
            for event in session:
                item_id=event[0]
                timestamp=event[1]
                if timestamp not in time_map:
                    time_map[timestamp]=int(round(timestamp-global_min_timestamp))#减去全局最小时间戳
                if item_id not in item_map:
                    iid= len(item_map)+1  #{itemid:1,itemid:2,....}
                    item_map[item_id]=iid

                interacted_set.add(item_map[item_id])#记录每个user 交互items
        unique_interactions_num += len(interacted_set)
        interacted[uid]=list(interacted_set)

    Density = unique_interactions_num / float(len(item_map) *len(user_map))
    print("item:",len(item_map),"user:",len(user_map),str(round(Density,5)))
    timestamp = str(datetime.datetime.now())
    config='\n'+timestamp+'||'+str(Max_SESSION_LEN)+'i:'+str(Min_SESSIONS)+'W||user_num:'+str(len(user_map))+'||item_num:'+str(len(item_map))+'||Density:'+str(round(Density,5))
    with open(cache_dir+'STATISTIC.txt', 'a+') as f:
        f.write(config + '\n')
    save_pickle((item_map,user_map,time_map),cache_dir+str(Max_SESSION_LEN)+'i/2item_user_time_mapped')
    save_pickle(interacted, cache_dir+str(Max_SESSION_LEN)+'i/2interacted')

    return user_map,item_map,time_map

def R_split_train_test():
    Data=load_pickle(cache_dir+str(Max_SESSION_LEN)+'i/3user_sessions')
    datas=Data['user_sessions']
    scale_timedlts=Data['scale_timedlts'][0]#min

    trainset = {}
    testset = {}
    validset = {}
    train_user_num = 0

    for key, val in datas.items():
        train_user_num += 1
        n = len(val)
        trainset[key] = val[:-2]
        validset[key] = val[-(WINDOWS_size + 1):-1]
        testset[key] = val[-WINDOWS_size:]

    traindata, train_user_num, train_samples_num = get_datas(trainset, scale_timedlts, state='train')
    validdata, valid_user_num, val_samples_num = get_datas(validset, scale_timedlts, state='val')
    testdata, test_user_num, test_samples_num = get_datas(testset, scale_timedlts, state='test')

    '''5 data for ours'''
    print(" train for our model----------")
    pickle_dict_our={}
    pickle_dict_our['trainset'] = traindata
    pickle_dict_our['validset'] = validdata
    pickle_dict_our['testset'] = testdata
    pickle_dict_our['global_timedlt'] = global_timedlt
    #pickle_dict_our['train_valid_test_sessions_num'] = [train_sessions_num, valid_sessions_num, test_sessions_num]
    pickle_dict_our['users-train_valid_test_samples'] = [train_user_num,train_samples_num,val_samples_num, test_samples_num]

    DATASET_OURS = cache_dir+str(Max_SESSION_LEN)+'i/5train_val_test'+str(WINDOWS_size)+'_W'
    save_pickle(pickle_dict_our, DATASET_OURS)

    timestamp = str(datetime.datetime.now())
    config = '\n' + timestamp + '\nAFTER:Max_SESSION_LEN and WINDOWS_size:[' + str(Max_SESSION_LEN) + '_i,' + str(
        int(WINDOWS_size)) + ']\nstate:R_split_train_test :last as test' + \
             '\nusers-train_valid_test_samples:[' + str(int(train_user_num)) + ',' + str(
        int(train_samples_num)) + ',' + str(int(val_samples_num)) + ',' + str(int(test_samples_num)) + ']'+'\nglobal_timedlt:'+str(int(len(global_timedlt)))
    print(config)
    with open(cache_dir + 'STATISTIC.txt', 'a+') as f:
        f.write(config + '\n')

def get_datas(dataset,scale_timedlts,state):
    user_num=0
    samples_num=0
    new_datas=[]
    for key, val in dataset.items():
        scale_timedlt=scale_timedlts[key]
        user_num+=1
        n = len(val)
        if n > WINDOWS_size:  # 若该user session数超过11个，则划分窗口，得到多条样本,每11作为一个样本,1-10作为历史，第11个作为预测
            a = n-WINDOWS_size
            for i in range(a):
                sess = val[i:i + WINDOWS_size]
                history_ses, timedlt, gt,max_session_len,max_session_num= get_sample(sess, scale_timedlt)
                sample = (key, history_ses, timedlt, gt,max_session_len,max_session_num)
                new_datas.append(sample)
                samples_num += 1
        else:
            history_ses, timedlt, gt ,max_session_len,max_session_num= get_sample(val, scale_timedlt)
            sample = (key, history_ses, timedlt, gt,max_session_len,max_session_num)
            new_datas.append(sample)
            samples_num += 1
    return new_datas,user_num,samples_num

def get_sample(sess,scale_timedlt):
    history_sess = sess[:-1]
    target_sess = sess[-1] #有1-10作为历史，第11个作为待预测
    time1 = history_sess[-1][-1][-1]  # 最后一个item的时间
    time2 = target_sess[0][1]  # 第一个item的时间
    dlt = math.ceil(abs(time2 - time1) / max(1,scale_timedlt))
    if dlt not in global_timedlt:
        global_timedlt[dlt]=len(global_timedlt)
    timedlt = global_timedlt[dlt]
    gt=[event[0] for event in target_sess]

    history=[[event[0] for event in session] for session in history_sess]
    max_session_len=np.max([len(session) for session in history])
    max_session_num=len(history)

    return history, timedlt, gt,max_session_len,max_session_num

def map_to_name():
    r_columns = ['item_id', 'title', 'genres']
    if "Amazon" in DATASET:
        name_df = pd.read_csv(Amazon_DATASET_FILE, sep=',', names=r_columns, engine='python', iterator=True)
    if "MovieLens" in DATASET:
        name_df = pd.read_csv(DATASET_DIR+Movie+'/movies.dat', sep='::', names=r_columns, engine='python', iterator=True)

    loop = True
    chunkSize = 10000
    chunks = []
    index = 0
    while loop:
        try:
            chunk = name_df.get_chunk(chunkSize)
            chunks.append(chunk)
            index += 1
        except StopIteration:
            loop = False
            print('StopIteration')
    print('combination')
    name_df = pd.concat(chunks, ignore_index=True)
    name_data={}
    item_map, user_map, _= load_pickle(cache_dir + str(Max_SESSION_LEN) + 'i/2item_user_time_mapped')
    for interaction in name_df.iterrows():
        item_id = int(interaction[1][0])
        title =interaction[1][1].split(',')[0] if interaction[1][1].find(',')>0 else interaction[1][1].split('(')[0]
        genre=interaction[1][2]
        if item_id in item_map:
            name_data[item_map[item_id]]=[title,genre]
            print(item_id,item_map[item_id],interaction,title)
    save_pickle(name_data,cache_dir + str(Max_SESSION_LEN) +'i/2map_to_name')

if __name__ == "__main__":

    Datas = Data_read()
    remove_unfrequent(Datas)
    split_sessions()
    R_split_train_test()
    #map_to_name()
















