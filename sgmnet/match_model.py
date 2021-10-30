import torch
import torch.nn as nn

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

        
def seeding(nn_index1,nn_index2,x1,x2,topk,match_score,confbar,nms_radius,use_mc=True,test=False):
    
    #apply mutual check before nms
    if use_mc:
        mask_not_mutual=nn_index2.gather(dim=-1,index=nn_index1)!=torch.arange(nn_index1.shape[1],device='cuda')
        match_score[mask_not_mutual]=-1
    #NMS
    pos_dismat1=((x1.norm(p=2,dim=-1)**2).unsqueeze_(-1)+(x1.norm(p=2,dim=-1)**2).unsqueeze_(-2)-2*(x1@x1.transpose(1,2))).abs_().sqrt_()
    x2=x2.gather(index=nn_index1.unsqueeze(-1).expand(-1,-1,2),dim=1)
    pos_dismat2=((x2.norm(p=2,dim=-1)**2).unsqueeze_(-1)+(x2.norm(p=2,dim=-1)**2).unsqueeze_(-2)-2*(x2@x2.transpose(1,2))).abs_().sqrt_()
    radius1, radius2 = nms_radius * pos_dismat1.mean(dim=(1,2),keepdim=True), nms_radius * pos_dismat2.mean(dim=(1,2),keepdim=True)
    nms_mask = (pos_dismat1 >= radius1) & (pos_dismat2 >= radius2)
    mask_not_local_max=(match_score.unsqueeze(-1)>=match_score.unsqueeze(-2))|nms_mask
    mask_not_local_max=~(mask_not_local_max.min(dim=-1).values)
    match_score[mask_not_local_max] = -1
 
    #confidence bar
    match_score[match_score<confbar]=-1
    mask_survive=match_score>0
    if test:
        topk=min(mask_survive.sum(dim=1)[0]+2,topk)
    _,topindex = torch.topk(match_score,topk,dim=-1)#b*k
    seed_index1,seed_index2=topindex,nn_index1.gather(index=topindex,dim=-1)
    return seed_index1,seed_index2



class PointCN(nn.Module):
    def __init__(self, channels,out_channels):
        nn.Module.__init__(self)
        self.shot_cut = nn.Conv1d(channels, out_channels, kernel_size=1)
        self.conv = nn.Sequential(
            nn.InstanceNorm1d(channels, eps=1e-3),
            nn.SyncBatchNorm(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.InstanceNorm1d(channels, eps=1e-3),
            nn.SyncBatchNorm(channels),
            nn.ReLU(),
            nn.Conv1d(channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        return self.conv(x) + self.shot_cut(x)


class attention_propagantion(nn.Module):

    def __init__(self,channel,head):
        nn.Module.__init__(self)
        self.head=head
        self.head_dim=channel//head
        self.query_filter,self.key_filter,self.value_filter=nn.Conv1d(channel,channel,kernel_size=1),nn.Conv1d(channel,channel,kernel_size=1),\
                                                            nn.Conv1d(channel,channel,kernel_size=1)
        self.mh_filter=nn.Conv1d(channel,channel,kernel_size=1)
        self.cat_filter=nn.Sequential(nn.Conv1d(2*channel,2*channel, kernel_size=1), nn.SyncBatchNorm(2*channel), nn.ReLU(),
                                      nn.Conv1d(2*channel, channel, kernel_size=1))

    def forward(self,desc1,desc2,weight_v=None):
        #desc1(q) attend to desc2(k,v)
        batch_size=desc1.shape[0]
        query,key,value=self.query_filter(desc1).view(batch_size,self.head,self.head_dim,-1),self.key_filter(desc2).view(batch_size,self.head,self.head_dim,-1),\
                        self.value_filter(desc2).view(batch_size,self.head,self.head_dim,-1)
        if weight_v is not None:
            value=value*weight_v.view(batch_size,1,1,-1)
        score=torch.softmax(torch.einsum('bhdn,bhdm->bhnm',query,key)/ self.head_dim ** 0.5,dim=-1)
        add_value=torch.einsum('bhnm,bhdm->bhdn',score,value).reshape(batch_size,self.head_dim*self.head,-1)
        add_value=self.mh_filter(add_value)
        desc1_new=desc1+self.cat_filter(torch.cat([desc1,add_value],dim=1))
        return desc1_new


class hybrid_block(nn.Module):
    def __init__(self,channel,head):
        nn.Module.__init__(self)
        self.head=head
        self.channel=channel
        self.attention_block_down = attention_propagantion(channel, head)
        self.cluster_filter=nn.Sequential(nn.Conv1d(2*channel,2*channel, kernel_size=1), nn.SyncBatchNorm(2*channel), nn.ReLU(),
                                         nn.Conv1d(2*channel, 2*channel, kernel_size=1))
        self.cross_filter=attention_propagantion(channel,head)
        self.confidence_filter=PointCN(2*channel,1)
        self.attention_block_self=attention_propagantion(channel,head)
        self.attention_block_up=attention_propagantion(channel,head)
        
    def forward(self,desc1,desc2,seed_index1,seed_index2):
        cluster1, cluster2 = desc1.gather(dim=-1, index=seed_index1.unsqueeze(1).expand(-1, self.channel, -1)), \
                             desc2.gather(dim=-1, index=seed_index2.unsqueeze(1).expand(-1, self.channel, -1))
        
        #pooling
        cluster1, cluster2 = self.attention_block_down(cluster1, desc1), self.attention_block_down(cluster2, desc2)
        concate_cluster=self.cluster_filter(torch.cat([cluster1,cluster2],dim=1))
        #filtering
        cluster1,cluster2=self.cross_filter(concate_cluster[:,:self.channel],concate_cluster[:,self.channel:]),\
                        self.cross_filter(concate_cluster[:,self.channel:],concate_cluster[:,:self.channel])
        cluster1,cluster2=self.attention_block_self(cluster1,cluster1),self.attention_block_self(cluster2,cluster2)
        #unpooling
        seed_weight=self.confidence_filter(torch.cat([cluster1,cluster2],dim=1))
        seed_weight=torch.sigmoid(seed_weight).squeeze(1)
        desc1_new,desc2_new=self.attention_block_up(desc1,cluster1,seed_weight),self.attention_block_up(desc2,cluster2,seed_weight)
        return desc1_new,desc2_new,seed_weight



class matcher(nn.Module):
    def __init__(self,config):
        nn.Module.__init__(self)
        self.seed_top_k=config.seed_top_k
        self.conf_bar=config.conf_bar
        self.seed_radius_coe=config.seed_radius_coe
        self.use_score_encoding=config.use_score_encoding
        self.detach_iter=config.detach_iter
        self.seedlayer=config.seedlayer
        self.layer_num=config.layer_num
        self.sink_iter=config.sink_iter

        self.position_encoder = nn.Sequential(nn.Conv1d(3, 32, kernel_size=1) if config.use_score_encoding else nn.Conv1d(2, 32, kernel_size=1), 
                                            nn.SyncBatchNorm(32),nn.ReLU(),
                                            nn.Conv1d(32, 64, kernel_size=1), nn.SyncBatchNorm(64),nn.ReLU(),
                                            nn.Conv1d(64, 128, kernel_size=1), nn.SyncBatchNorm(128),nn.ReLU(),
                                            nn.Conv1d(128, 256, kernel_size=1), nn.SyncBatchNorm(256),nn.ReLU(),
                                            nn.Conv1d(256, config.net_channels, kernel_size=1))
     
        
        self.hybrid_block=nn.Sequential(*[hybrid_block(config.net_channels, config.head) for _ in range(config.layer_num)])
        self.final_project = nn.Conv1d(config.net_channels, config.net_channels, kernel_size=1)
        self.dustbin=nn.Parameter(torch.tensor(1.5,dtype=torch.float32))
        
        #if reseeding
        if len(config.seedlayer)!=1:
            self.mid_dustbin=nn.ParameterDict({str(i):nn.Parameter(torch.tensor(2,dtype=torch.float32)) for i in config.seedlayer[1:]})
            self.mid_final_project = nn.Conv1d(config.net_channels, config.net_channels, kernel_size=1)
       
    def forward(self,data,test_mode=True):
        x1, x2, desc1, desc2 = data['x1'][:,:,:2], data['x2'][:,:,:2], data['desc1'], data['desc2']
        desc1, desc2 = torch.nn.functional.normalize(desc1,dim=-1), torch.nn.functional.normalize(desc2,dim=-1)
        if test_mode:
            encode_x1,encode_x2=data['x1'],data['x2']
        else:
            encode_x1,encode_x2=data['aug_x1'], data['aug_x2']
    
        #preparation
        desc_dismat=(2-2*torch.matmul(desc1,desc2.transpose(1,2))).sqrt_()
        values,nn_index=torch.topk(desc_dismat,k=2,largest=False,dim=-1,sorted=True)
        nn_index2=torch.min(desc_dismat,dim=1).indices.squeeze(1)
        inverse_ratio_score,nn_index1=values[:,:,1]/values[:,:,0],nn_index[:,:,0]#get inverse score
   
        #initial seeding
        seed_index1,seed_index2=seeding(nn_index1,nn_index2,x1,x2,self.seed_top_k[0],inverse_ratio_score,self.conf_bar[0],\
                                self.seed_radius_coe,test=test_mode) 

        #position encoding
        desc1,desc2=desc1.transpose(1,2),desc2.transpose(1,2)   
        if not self.use_score_encoding:
            encode_x1,encode_x2=encode_x1[:,:,:2],encode_x2[:,:,:2]
        encode_x1,encode_x2=encode_x1.transpose(1,2),encode_x2.transpose(1,2)
        x1_pos_embedding, x2_pos_embedding = self.position_encoder(encode_x1), self.position_encoder(encode_x2)
        aug_desc1, aug_desc2 = x1_pos_embedding + desc1, x2_pos_embedding + desc2
      
        seed_weight_tower,mid_p_tower,seed_index_tower,nn_index_tower=[],[],[],[]
        seed_index_tower.append(torch.stack([seed_index1, seed_index2],dim=-1))
        nn_index_tower.append(nn_index1)

        seed_para_index=0
        for i in range(self.layer_num):
            #mid seeding
            if i in self.seedlayer and i!= 0:
                seed_para_index+=1
                aug_desc1,aug_desc2=self.mid_final_project(aug_desc1),self.mid_final_project(aug_desc2)
                M=torch.matmul(aug_desc1.transpose(1,2),aug_desc2)
                p=sink_algorithm(M,self.mid_dustbin[str(i)],self.sink_iter[seed_para_index-1])
                mid_p_tower.append(p)
                #rematching with p
                values,nn_index=torch.topk(p[:,:-1,:-1],k=1,dim=-1)
                nn_index2=torch.max(p[:,:-1,:-1],dim=1).indices.squeeze(1)
                p_match_score,nn_index1=values[:,:,0],nn_index[:,:,0]
                #reseeding
                seed_index1, seed_index2 = seeding(nn_index1,nn_index2,x1,x2,self.seed_top_k[seed_para_index],p_match_score,\
                                                    self.conf_bar[seed_para_index],self.seed_radius_coe,test=test_mode)
                seed_index_tower.append(torch.stack([seed_index1, seed_index2],dim=-1)), nn_index_tower.append(nn_index1)
                if not test_mode and data['step']<self.detach_iter:
                    aug_desc1,aug_desc2=aug_desc1.detach(),aug_desc2.detach()

            aug_desc1, aug_desc2,seed_weight=self.hybrid_block[i](aug_desc1, aug_desc2,seed_index1,seed_index2)
            seed_weight_tower.append(seed_weight)
        
    
        aug_desc1,aug_desc2 = self.final_project(aug_desc1), self.final_project(aug_desc2)
        cmat = torch.matmul(aug_desc1.transpose(1, 2), aug_desc2)
        p = sink_algorithm(cmat, self.dustbin,self.sink_iter[-1])
        #seed_weight_tower: l*b*k
        #seed_index_tower: l*b*k*2
        #nn_index_tower: seed_l*b
        return {'p':p,'seed_conf':seed_weight_tower,'seed_index':seed_index_tower,'mid_p':mid_p_tower,'nn_index':nn_index_tower}


