#!/PycharmProjects/env python
#-*- coding:utf-8 -*-
#import pandas as pd
import numpy as np
import pickle
import os
import argparse
import logging
import time
import torch
import matplotlib.pyplot as plt
from LST_model import LST
import math
from util import *
import matplotlib.ticker as ticker
from dataload import *
from torch.optim.lr_scheduler import LambdaLR
import torchvision
from torch.utils.data import DataLoader,Dataset
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_batch(data,batchsize,state):
    total_samples = len(data)
    data = np.array(data,dtype=object)
    num_batchs = int(total_samples / batchsize) + 1
    for batchID in range(num_batchs):
        start = batchID * batchsize
        end = start + batchsize
        if batchID == num_batchs - 1:
            if start < total_samples:
                end = total_samples
            else:
                break
        batch_u =data[start:end,0].tolist()
        batch_session = data[start:end,1].tolist()
        batch_timedlt = data[start:end,2].tolist()
        batch_gt = data[start:end,3].tolist()
        max_session_len=np.max(data[start:end,4])
        max_session_num=np.max(data[start:end,5])
        gt_lens = np.array([len(item) for item in batch_gt])
        max_gt_len=gt_lens.max()

        batch_pad_session = [[[0]*max_session_len]*(max_session_num -len(sessions))+[[0] * (max_session_len - len(session))+session for session in sessions] for sessions in batch_session]

        if state=='train':
            padding_gts,batch_neg=f.get_neg(batch_u,batch_gt,max_gt_len)
            yield batch_u, batch_pad_session, batch_timedlt, padding_gts,batch_neg
        else:
            yield batch_u, batch_pad_session, batch_timedlt, batch_gt


def evaluate(model, data, batchsize,state):
    model.eval()
    if state=='test':
        top_k = [5, 10, 20, 30]
    else:
        top_k=[5]
    all_Recall,all_Precision,all_HR ,all_NDCG,all_MRR,all_hits = [0.0] * len(top_k),[0.0]*len(top_k),[0.0]*len(top_k),[0.0]*len(top_k),[0.0]*len(top_k),[0.0]*len(top_k)
    all_count=0
    with torch.no_grad():
        for i, x in enumerate(get_batch(data, batchsize,state='eva')):
            batch_u, batch_session, batch_timedlt, batch_gt = x
            all_count+=len(batch_u)
            topk_rank= model(batch_u,batch_session,batch_timedlt, None)
            topk_rank_list=topk_rank.cuda().data.cpu().numpy()
            for i in range(len(top_k)):
                hr,recall,precision,ndcg,mrr,hits=metrics(batch_gt, topk_rank_list[:, :top_k[i]],top_k[i])
                all_HR[i] += hr
                all_Recall[i] +=recall
                all_Precision[i]+=precision
                all_NDCG[i] += ndcg
                all_MRR[i] += mrr
                all_hits[i]+=hits

    HR_k = [x / all_count for x in all_HR]
    Recall_k = [x / all_count for x in all_Recall]
    Precision_k = [x / all_count for x in all_Precision]
    NDCG_k = [x / all_count for x in all_NDCG]
    MRR_k = [x / all_count for x in all_MRR]
    p=Precision_k
    r=Recall_k
    F1=[2*p[j]*r[j]/(p[j]+r[j]) if (p[j]+r[j])>0 else 0 for j in range(len(top_k))]
    return HR_k,Recall_k,Precision_k,NDCG_k,MRR_k,F1

def main(config,f):
    is_train, is_test, reload_model = True, True, False

    model_save_path=cache_dir + str(SESSION_LEN) + 'i/' + str(WINDOWS_size) + 'w_model_' + str(
                    int(config.d_model)) + 'd' + str(config.blocks) + 'block' + str(config.h) + 'head_parameter.pt'
    model = LST(config)
    testdata=f.testset
    total_test=len(testdata)
    trainset=f.trainset
    total_train=len(trainset)
    validset=f.validset
    total_valid=len(validset)

    message = "\n------------------------------------------------------------------------\n"
    if is_train:
        if reload_model:
            model.load_state_dict(torch.load(model_save_path))
        message += "\nCONFIG: " + str(config) + "\ntrain lens: " + str(total_train) + "|| valid lens: " + str(
            total_valid)

        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, betas=(0.9, 0.98), eps=1e-8, weight_decay=config.weight_decay)
        best_val_F1_5 = float('-inf')
        bad=0

        for epoch in range(config.Epochs):
            model.train()
            start_time = time.time()
            epoch_loss=0
            np.random.shuffle(trainset)
            for i, x in enumerate(get_batch(trainset,config.batch_size,state='train')):
                batch_u, batch_session, batch_timedlt, batch_gt, batch_neg = x

                gts = torch.from_numpy(batch_gt).type(torch.LongTensor).to(device)
                negs = torch.from_numpy(batch_neg).type(torch.LongTensor).to(device)
                batch_to_predict = torch.cat((gts, negs), 1)

                predict_scores = model(batch_u, batch_session, batch_timedlt, batch_to_predict)

                (gt_scores, neg_scores) = torch.split(predict_scores, [gts.size(1), negs.size(1)],
                                                      dim=1)  # [gts.size(1), negs.size(1)]


                loss = -torch.sum(torch.log(torch.sigmoid(gt_scores - neg_scores) + 1e-24))
                epoch_loss += loss.item()
                optimizer.zero_grad()  # 前一步梯度的损失清零
                loss.backward()
                optimizer.step()  # 优化

            train_end_time = time.time()
            train_epoch_mins, train_epoch_secs = epoch_time(start_time, train_end_time)
            train_message="\ntrain|| Epoch:\t" + str(epoch + 1) +"|Time: "+str(train_epoch_mins)+"m"+str(train_epoch_secs )+ "s|\tTrain epoch Loss" + str(
                        round(epoch_loss, 6))
            message+=train_message
            print(train_message)
            if (epoch+1)%5== 0:
                HR_5,Recall_5,Precision_5,NDCG_5,MRR_5,F1=evaluate(model, testdata, config.batch_size, state='val')
                current_val_F1_5=F1[0]
                valid_message= "\n------------Valid Ranking---------\nK=5" + "\t" + str(round(HR_5[0], 6)) + "\t" + str(
                    round(Recall_5[0], 6)) + "\t" + str(round(Precision_5[0], 6)) + "\t" +str(round(NDCG_5[0],6)) + "\t" + str(round(MRR_5[0],6))+"\t"+str(round(F1[0],6))
                message+=valid_message
                print(valid_message)
                if best_val_F1_5 < current_val_F1_5:
                    best_val_F1_5 = current_val_F1_5
                    torch.save(model.state_dict(), model_save_path)  # 保存模型参数
                else:
                    bad+=1
            if bad>30:
                break

    if is_test:
        model.load_state_dict(torch.load(model_save_path))
        model.eval()
        HR_k,Recall_k,Precision_k, NDCG_k, MRR_k,F1 = evaluate(model, testdata, config.batch_size, state='test')
        test_message= "\n------------Test Ranking---------\n" \
                   " || test lens: " + str(total_test)+"\n--TOP-K\tHR_k\tRecall_k\tPrecision_k\tNDCG_k\tMRR_k\tF1"
        top_k = [5, 10, 20, 30]
        for j in range(len(top_k)):
            test_message += "\nK=" + str(top_k[j]) + "\t" + str(round(HR_k[j], 6)) + "\t" + str(round(
                Recall_k[j], 6)) + "\t"+str(round(Precision_k[j],6))+"\t" + str(round(NDCG_k[j], 6)) + "\t" + str(round(MRR_k[j], 6))+"\t"+str(round(F1[j],6))  # format(top_k[j], '.6f')
        print(test_message)
        message+=test_message
        
    f.log_config(message)

DIR="E:/CODE/DATASETS"
Amanzon="/Amazon_Video"
Movie="/MovieLens1m"
Lastfm="/Lastfm"
DATASET=Lastfm
cache_dir=DIR+DATASET+'/LST/'
SESSION_LEN,WINDOWS_size=20,6

if __name__ == '__main__':
    f = DataHandler(cache_dir,SESSION_LEN,WINDOWS_size)
    num_items, num_users, num_global_timedlt = f.num_items, f.num_users, len(f.global_timedlt)

    parser = argparse.ArgumentParser()
    parser.add_argument('-- Experiment ', type=str, default='HATSRec')
    parser.add_argument('-- DATASET ', type=str, default=DATASET)
    parser.add_argument('-- Loss_opt ', type=str, default='BPR')
    parser.add_argument('--Epochs', type=int, default=250)
    parser.add_argument('--batch_size', type=int, default=128)#movie 256, lastfm:512, VIDEO:64 book64[64,128,256,512]
    parser.add_argument('--learning_rate', type=float, default=0.01)#movie0.001,lastfm:0.001, video:0.01 Book:0.01[0.1,0.01,0.001,0.0001]
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=0.001) #movie0.001,lastfm:0.001 , video:0.01[1,0.1,0.01,0.001,0.0001]
    parser.add_argument('--inter_len', type=int, default=WINDOWS_size)
    parser.add_argument('--intra_len', type=int, default=SESSION_LEN)
    parser.add_argument('--blocks', type=int, default=1)
    parser.add_argument('--h', type=int, default=8)
    parser.add_argument('--d_model', type=int, default=32)
    parser.add_argument('--session_pool_type',type=str,default='mean')
    parser.add_argument('--num_items', type=int, default=num_items)
    parser.add_argument('--num_users', type=int, default=num_users)
    parser.add_argument('--num_timedlt', type=int, default=num_global_timedlt)
    config = parser.parse_args()
    main(config,f)

