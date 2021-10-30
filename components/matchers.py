import torch
import numpy as np
import os
from collections import OrderedDict,namedtuple
import sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from sgmnet import matcher as SGM_Model
from superglue import matcher as SG_Model
from utils import evaluation_utils

class GNN_Matcher(object):

    def __init__(self,config,model_name):
        assert model_name=='SGM' or model_name=='SG'

        config=namedtuple('config',config.keys())(*config.values())
        self.p_th=config.p_th
        self.model = SGM_Model(config) if model_name=='SGM' else SG_Model(config) 
        self.model.cuda(),self.model.eval()
        checkpoint = torch.load(os.path.join(config.model_dir, 'model_best.pth'))
        #for ddp model
        if list(checkpoint['state_dict'].items())[0][0].split('.')[0]=='module':
            new_stat_dict=OrderedDict()
            for key,value in checkpoint['state_dict'].items():
                new_stat_dict[key[7:]]=value
            checkpoint['state_dict']=new_stat_dict
        self.model.load_state_dict(checkpoint['state_dict'])

    def run(self,test_data):
        norm_x1,norm_x2=evaluation_utils.normalize_size(test_data['x1'][:,:2],test_data['size1']),\
                                                    evaluation_utils.normalize_size(test_data['x2'][:,:2],test_data['size2'])
        x1,x2=np.concatenate([norm_x1,test_data['x1'][:,2,np.newaxis]],axis=-1),np.concatenate([norm_x2,test_data['x2'][:,2,np.newaxis]],axis=-1)
        feed_data={'x1':torch.from_numpy(x1[np.newaxis]).cuda().float(),
                   'x2':torch.from_numpy(x2[np.newaxis]).cuda().float(),
                   'desc1':torch.from_numpy(test_data['desc1'][np.newaxis]).cuda().float(),
                   'desc2':torch.from_numpy(test_data['desc2'][np.newaxis]).cuda().float()}
        with torch.no_grad():
            res=self.model(feed_data,test_mode=True)
            p=res['p']
        index1,index2=self.match_p(p[0,:-1,:-1])
        corr1,corr2=test_data['x1'][:,:2][index1.cpu()],test_data['x2'][:,:2][index2.cpu()]
        if len(corr1.shape)==1:
            corr1,corr2=corr1[np.newaxis],corr2[np.newaxis]
        return corr1,corr2
    
    def match_p(self,p):#p N*M
        score,index=torch.topk(p,k=1,dim=-1)
        _,index2=torch.topk(p,k=1,dim=-2)
        mask_th,index,index2=score[:,0]>self.p_th,index[:,0],index2.squeeze(0)
        mask_mc=index2[index] == torch.arange(len(p)).cuda()
        mask=mask_th&mask_mc
        index1,index2=torch.nonzero(mask).squeeze(1),index[mask]
        return index1,index2


class NN_Matcher(object):

    def __init__(self,config):
        config=namedtuple('config',config.keys())(*config.values())
        self.mutual_check=config.mutual_check
        self.ratio_th=config.ratio_th

    def run(self,test_data):
        desc1,desc2,x1,x2=test_data['desc1'],test_data['desc2'],test_data['x1'],test_data['x2']
        desc_mat=np.sqrt(abs((desc1**2).sum(-1)[:,np.newaxis]+(desc2**2).sum(-1)[np.newaxis]-2*desc1@desc2.T))
        nn_index=np.argpartition(desc_mat,kth=(1,2),axis=-1)
        dis_value12=np.take_along_axis(desc_mat,nn_index, axis=-1)
        ratio_score=dis_value12[:,0]/dis_value12[:,1]
        nn_index1=nn_index[:,0]
        nn_index2=np.argmin(desc_mat,axis=0)
        mask_ratio,mask_mutual=ratio_score<self.ratio_th,np.arange(len(x1))==nn_index2[nn_index1]
        corr1,corr2=x1[:,:2],x2[:,:2][nn_index1]
        if self.mutual_check:
            mask=mask_ratio&mask_mutual
        else:
            mask=mask_ratio
        corr1,corr2=corr1[mask],corr2[mask]
        return corr1,corr2




