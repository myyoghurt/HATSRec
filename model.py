#!/PycharmProjects/env python
#-*- coding:utf-8 -*-
#!/PycharmProjects/env python
#-*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from attention_model import Sublayers,PositionalEncoding,Attention,LayerNorm
from util import pool_max, pool_avg,pool_sum
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class LST(nn.Module):
    def __init__(self, model_args):
        super(LST, self).__init__()
        self.args = model_args
        self.device = device
        d_model = self.args.d_model
        dropout = self.args.dropout
        self.padding_value=0
        # user and item embeddings
        self.item_embeddings = nn.Embedding(self.args.num_items, d_model, padding_idx=self.padding_value).to(self.device)
        self.user_embeddings = nn.Embedding(self.args.num_users, d_model).to(self.device)
        self.timedlt_embeddings = nn.Embedding(self.args.num_timedlt, d_model).to(self.device)

        self.pool ={'mean': pool_avg, 'max': pool_max,'sum':pool_sum}[self.args.session_pool_type]
        self.intra_attention = Attention(d_model, dropout, self.device)
        self.pos_embedding = PositionalEncoding(d_model,dropout).to(self.device)
        self.sublayers =Sublayers(d_model, self.args.h, 4*d_model, dropout,self.device)
        self.inter_fuse_attetion =nn.Linear(d_model,d_model,bias=True).to(self.device)

        self.gate_Wl=Variable(torch.zeros(d_model, d_model).type(torch.FloatTensor), requires_grad=True).to(self.device)
        self.gate_Wl = torch.nn.init.xavier_uniform_(self.gate_Wl)
        self.gate_Ws =Variable(torch.zeros(d_model, d_model).type(torch.FloatTensor), requires_grad=True).to(self.device)
        self.gate_Ws = torch.nn.init.xavier_uniform_(self.gate_Ws)
        self.gate_Wt=Variable(torch.zeros(d_model, d_model).type(torch.FloatTensor), requires_grad=True).to(self.device)
        self.gate_Wt = torch.nn.init.xavier_uniform_(self.gate_Wt)
        self.gate_bias =Variable(torch.zeros(1, d_model).type(torch.FloatTensor), requires_grad=True).to(self.device)
        self.gate_bias = torch.nn.init.xavier_uniform_(self.gate_bias)

    def forward(self,batch_us,batch_sessions, batch_timedlt, batch_to_predict):
        seq=torch.from_numpy(np.array(batch_sessions)).type(torch.LongTensor).to(self.device)
        E_embs = self.item_embeddings(seq)  # [batch_size , sessions_len , items , item_d_model]
        U_embs = self.user_embeddings(torch.from_numpy(np.array(batch_us)).type(torch.LongTensor).to(self.device))
        T_embs=self.timedlt_embeddings(torch.from_numpy(np.array(batch_timedlt)).type(torch.LongTensor).to(self.device))

        batchsize = E_embs.shape[0]
        inter_len = E_embs.shape[1]
        intra_len=E_embs.shape[2]

        padding_mask= (seq!=self.padding).view(-1,1,intra_len)
        E_hat,intra_weights = self.intra_attention(E_embs.view(-1, intra_len, self.args.d_model),mask=padding_mask)#[batchsize*sessions,items,d_model]
        Z=self.pool(E_hat,dim=1).view(batchsize, inter_len,self.args.d_model) #[batchsize, sessions,d_model]

        u_short = Z[:,-1,:]

        padding_future_mask = self.make_std_mask(seq[:, :, -1])
        S = self.pos_embedding(Z)
        for i in range(self.args.blocks):
            S,inter_weights=self.sublayers(S,mask=padding_future_mask)#[batchsize, sessions,d_model]

        omega=self.inter_fuse_attetion(S)
        omega=F.relu(omega)
        inter_fuse_weights = F.softmax(torch.bmm(omega, U_embs.unsqueeze(-1)).masked_fill(seq[:, :, -1:]==0,-1e9), dim=1)  # [batchsize,sessions,1]
        u_long = torch.sum(S*inter_fuse_weights, dim=1)  # [batchsize,d]

        long = torch.mm(u_long, self.gate_Wl)
        short = torch.mm(u_short, self.gate_Ws)
        time=torch.mm(T_embs,self.gate_Wt)
        g =torch.sigmoid(long+short+time+self.gate_bias)
        U = g * u_short + (1 - g) * u_long  #[batchsize,d_model]

        if batch_to_predict is None:
            item_embs=self.item_embeddings.weight.data
            score =F.sigmoid(torch.mm(U, item_embs.t()))
            _,R=torch.topk(score, k=31, dim=1)

        else:
            item_embs = self.item_embeddings(batch_to_predict.type(torch.LongTensor).to(self.device))
            R = F.sigmoid(torch.squeeze(torch.einsum('bij,bjk->bik', item_embs,torch.unsqueeze(U, 2)))) # [batch_size,items,d_model]*[batch_size,d_model,1]

        return R#,inter_weights,inter_fuse_weights.transpose(2,1),intra_weights

    def make_std_mask(self,seq):
        "Create a mask to hide padding and future "
        seq_len=seq.size(-1)
        seq_mask = (seq != self.padding_value).unsqueeze(-2)
        future_mask=np.tril(np.ones((1,seq_len, seq_len)), k=0).astype('uint8')
        seq_mask = seq_mask & (torch.from_numpy(future_mask) == 1).type_as(seq_mask.data)
        return seq_mask

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                n = m.in_features
                y = 1.0 / np.sqrt(n)
                m.weight.data.normal_(-y, y)
                m.bias.data.zero_()
    #         elif isinstance(m, nn.Embedding):
    #             m.weight.data.normal_(0.0, 1.0)












