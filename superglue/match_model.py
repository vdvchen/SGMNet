import torch
import torch.nn as nn
import time


eps=1e-8

def sinkhorn(M,r,c,iteration):
    p = torch.softmax(M, dim=-1)
    u = torch.ones_like(r)
    v = torch.ones_like(c)
    for _ in range(iteration):
        u = r / ((p * v.unsqueeze(-2)).sum(-1) + eps)
        v = c / ((p * u.unsqueeze(-1)).sum(-2) + eps)
    p = p * u.unsqueeze(-1) * v.unsqueeze(-2)
    return p

def sink_algorithm(M,dustbin,iteration):
    M = torch.cat([M, dustbin.expand([M.shape[0], M.shape[1], 1])], dim=-1)
    M = torch.cat([M, dustbin.expand([M.shape[0], 1, M.shape[2]])], dim=-2)
    r = torch.ones([M.shape[0], M.shape[1] - 1],device='cuda')
    r = torch.cat([r, torch.ones([M.shape[0], 1],device='cuda') * M.shape[1]], dim=-1)
    c = torch.ones([M.shape[0], M.shape[2] - 1],device='cuda')
    c = torch.cat([c, torch.ones([M.shape[0], 1],device='cuda') * M.shape[2]], dim=-1)
    p=sinkhorn(M,r,c,iteration)
    return p


class attention_block(nn.Module):
    def __init__(self,channels,head,type):
        assert type=='self' or type=='cross','invalid attention type'
        nn.Module.__init__(self)
        self.head=head
        self.type=type
        self.head_dim=channels//head
        self.query_filter=nn.Conv1d(channels, channels, kernel_size=1)
        self.key_filter=nn.Conv1d(channels,channels,kernel_size=1)
        self.value_filter=nn.Conv1d(channels,channels,kernel_size=1)
        self.attention_filter=nn.Sequential(nn.Conv1d(2*channels,2*channels, kernel_size=1),nn.SyncBatchNorm(2*channels), nn.ReLU(),
                                             nn.Conv1d(2*channels, channels, kernel_size=1))
        self.mh_filter=nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self,fea1,fea2):
        batch_size,n,m=fea1.shape[0],fea1.shape[2],fea2.shape[2]
        query1, key1, value1 = self.query_filter(fea1).view(batch_size,self.head_dim,self.head,-1), self.key_filter(fea1).view(batch_size,self.head_dim,self.head,-1), \
                               self.value_filter(fea1).view(batch_size,self.head_dim,self.head,-1)
        query2, key2, value2 = self.query_filter(fea2).view(batch_size,self.head_dim,self.head,-1), self.key_filter(fea2).view(batch_size,self.head_dim,self.head,-1), \
                               self.value_filter(fea2).view(batch_size,self.head_dim,self.head,-1)
        if(self.type=='self'):
            score1,score2=torch.softmax(torch.einsum('bdhn,bdhm->bhnm',query1,key1)/self.head_dim**0.5,dim=-1),\
                          torch.softmax(torch.einsum('bdhn,bdhm->bhnm',query2,key2)/self.head_dim**0.5,dim=-1)
            add_value1, add_value2 = torch.einsum('bhnm,bdhm->bdhn', score1, value1), torch.einsum('bhnm,bdhm->bdhn',score2, value2)
        else:
            score1,score2 = torch.softmax(torch.einsum('bdhn,bdhm->bhnm', query1, key2) / self.head_dim ** 0.5,dim=-1), \
                            torch.softmax(torch.einsum('bdhn,bdhm->bhnm', query2, key1) / self.head_dim ** 0.5, dim=-1)
            add_value1, add_value2 =torch.einsum('bhnm,bdhm->bdhn',score1,value2),torch.einsum('bhnm,bdhm->bdhn',score2,value1)
        add_value1,add_value2=self.mh_filter(add_value1.contiguous().view(batch_size,self.head*self.head_dim,n)),self.mh_filter(add_value2.contiguous().view(batch_size,self.head*self.head_dim,m))
        fea11, fea22 = torch.cat([fea1, add_value1], dim=1), torch.cat([fea2, add_value2], dim=1)
        fea1, fea2 = fea1+self.attention_filter(fea11), fea2+self.attention_filter(fea22)
     
        return fea1,fea2


class matcher(nn.Module):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.use_score_encoding=config.use_score_encoding
        self.layer_num=config.layer_num
        self.sink_iter=config.sink_iter
        self.position_encoder = nn.Sequential(nn.Conv1d(3, 32, kernel_size=1) if config.use_score_encoding else nn.Conv1d(2, 32, kernel_size=1), 
                                              nn.SyncBatchNorm(32), nn.ReLU(),
                                              nn.Conv1d(32, 64, kernel_size=1), nn.SyncBatchNorm(64),nn.ReLU(),
                                              nn.Conv1d(64, 128, kernel_size=1), nn.SyncBatchNorm(128), nn.ReLU(),
                                              nn.Conv1d(128, 256, kernel_size=1), nn.SyncBatchNorm(256), nn.ReLU(),
                                              nn.Conv1d(256, config.net_channels, kernel_size=1))
       
        self.dustbin=nn.Parameter(torch.tensor(1,dtype=torch.float32,device='cuda'))
        self.self_attention_block=nn.Sequential(*[attention_block(config.net_channels,config.head,'self') for _ in range(config.layer_num)])
        self.cross_attention_block=nn.Sequential(*[attention_block(config.net_channels,config.head,'cross') for _ in range(config.layer_num)])
        self.final_project=nn.Conv1d(config.net_channels, config.net_channels, kernel_size=1)

    def forward(self,data,test_mode=True):
        desc1, desc2 = data['desc1'], data['desc2']
        desc1, desc2 = torch.nn.functional.normalize(desc1,dim=-1), torch.nn.functional.normalize(desc2,dim=-1)
        desc1,desc2=desc1.transpose(1,2),desc2.transpose(1,2)   
        if test_mode:
            encode_x1,encode_x2=data['x1'],data['x2']
        else:
            encode_x1,encode_x2=data['aug_x1'], data['aug_x2']
        if not self.use_score_encoding:
            encode_x1,encode_x2=encode_x1[:,:,:2],encode_x2[:,:,:2]

        encode_x1,encode_x2=encode_x1.transpose(1,2),encode_x2.transpose(1,2)

        x1_pos_embedding, x2_pos_embedding = self.position_encoder(encode_x1), self.position_encoder(encode_x2)
        aug_desc1, aug_desc2 = x1_pos_embedding + desc1, x2_pos_embedding+desc2
        for i in range(self.layer_num):
            aug_desc1,aug_desc2=self.self_attention_block[i](aug_desc1,aug_desc2)
            aug_desc1,aug_desc2=self.cross_attention_block[i](aug_desc1,aug_desc2)

        aug_desc1,aug_desc2=self.final_project(aug_desc1),self.final_project(aug_desc2)
        desc_mat = torch.matmul(aug_desc1.transpose(1, 2), aug_desc2)
        p = sink_algorithm(desc_mat, self.dustbin,self.sink_iter[0])
        return {'p':p}


