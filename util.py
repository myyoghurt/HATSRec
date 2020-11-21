
import pickle
import torch
import datetime
import json
import math
import numpy as np

def save_pickle(obj, file_name):
    with open(file_name + '.pickle', 'wb') as g:
        pickle.dump(obj, g)

def load_pickle(file_name):
    with open(file_name + '.pickle', 'rb') as g:
            return pickle.load(g)
def data_save_to_json(obj, file_name):
    # obj.to_csv(file_name+'.csv')
    jsObj = json.dumps(obj)
    fileObject = open(file_name + '.json', 'w')
    fileObject.write(jsObj)
    fileObject.close()
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def process_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_hours=int(elapsed_time / 3600)
    elapsed_mins = int((elapsed_time -(elapsed_hours*60))/60)
    return elapsed_hours,elapsed_mins

def pool_max(tensor, dim):
    return torch.max(tensor, dim)[0]

def pool_avg(tensor, dim):
    return torch.mean(tensor, dim)
def pool_sum(tensor,dim):
    return torch.sum(tensor, dim)

'''for evaluating'''

def metrics(act_items, topk_items_matrix,topk):
    sum_mrr = 0.0
    sum_ndcg=0.0
    sum_recall = 0.0
    sum_hr=0.0
    sum_precision=0.0
    sum_hits=0
    num_sample = len(act_items)
    for i in range(num_sample):
        predict=topk_items_matrix[i]
        act=act_items[i]
        dcg_k=0.0
        hits = 0
        mrr=0.0
        relevant=np.zeros_like(predict)
        for j in range(len(predict)):
            if predict[j] in act:
                hits+=1.0
                mrr+= hits/(j + 1)
                dcg_k+=1.0/math.log(j+2,2)
                relevant[j]=1.0
        if hits>0:
            sum_mrr+=mrr/len(set(act))
            sum_precision +=hits/topk
            sum_recall += hits/len(set(act))
            sum_hits +=hits
            sum_hr+=1.0
            rel=sorted(relevant,reverse=True)
            idcg = sum([rel[o] / math.log(o + 2, 2) for o in range(topk)])
            sum_ndcg += dcg_k / idcg

    return sum_hr,sum_recall,sum_precision,sum_ndcg,sum_mrr,sum_hits
