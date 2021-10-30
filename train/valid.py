import torch
import numpy as np
import cv2
import os
from loss import batch_episym
from tqdm import tqdm

import sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from utils import evaluation_utils,train_utils


def valid(valid_loader, model,match_loss, config,model_config):
    model.eval()
    loader_iter = iter(valid_loader)
    num_pair = 0
    total_loss,total_acc_corr,total_acc_incorr=0,0,0
    total_precision,total_recall=torch.zeros(model_config.layer_num ,device='cuda'),\
                                 torch.zeros(model_config.layer_num ,device='cuda')
    total_acc_mid=torch.zeros(len(model_config.seedlayer)-1,device='cuda')

    with torch.no_grad():
        if config.local_rank==0:
            loader_iter=tqdm(loader_iter)
            print('validating...')
        for test_data in loader_iter:
            num_pair+= 1
            test_data = train_utils.tocuda(test_data)
            res= model(test_data)
            loss_res=match_loss.run(test_data,res)
           
            total_acc_corr+=loss_res['acc_corr']
            total_acc_incorr+=loss_res['acc_incorr']
            total_loss+=loss_res['total_loss']

            if config.model_name=='SGM':
                total_acc_mid+=loss_res['mid_acc_corr']
                total_precision,total_recall=total_precision+loss_res['pre_seed_conf'],total_recall+loss_res['recall_seed_conf']
                
        total_acc_corr/=num_pair
        total_acc_incorr /= num_pair
        total_precision/=num_pair
        total_recall/=num_pair
        total_acc_mid/=num_pair

        #apply tensor reduction
        total_loss,total_acc_corr,total_acc_incorr,total_precision,total_recall,total_acc_mid=train_utils.reduce_tensor(total_loss,'sum'),\
                        train_utils.reduce_tensor(total_acc_corr,'mean'),train_utils.reduce_tensor(total_acc_incorr,'mean'),\
                        train_utils.reduce_tensor(total_precision,'mean'),train_utils.reduce_tensor(total_recall,'mean'),train_utils.reduce_tensor(total_acc_mid,'mean')
    model.train()
    return total_loss,total_acc_corr,total_acc_incorr,total_precision,total_recall,total_acc_mid



def dump_train_vis(res,data,step,config):
    #batch matching
    p=res['p'][:,:-1,:-1]
    score,index1=torch.max(p,dim=-1)
    _,index2=torch.max(p,dim=-2)
    mask_th=score>0.2
    mask_mc=index2.gather(index=index1,dim=1) == torch.arange(len(p[0])).cuda()[None]
    mask_p=mask_th&mask_mc#B*N

    corr1,corr2=data['x1'],data['x2'].gather(index=index1[:,:,None].expand(-1,-1,2),dim=1)
    corr1_kpt,corr2_kpt=data['kpt1'],data['kpt2'].gather(index=index1[:,:,None].expand(-1,-1,2),dim=1)
    epi_dis=batch_episym(corr1,corr2,data['e_gt'])
    mask_inlier=epi_dis<config.inlier_th#B*N

    #dump vis
    for cur_mask_p,cur_mask_inlier,cur_corr1,cur_corr2,img_path1,img_path2 in zip(mask_p,mask_inlier,corr1_kpt,corr2_kpt,data['img_path1'],data['img_path2']):
        img1,img2=cv2.imread(img_path1),cv2.imread(img_path2)
        dis_play=evaluation_utils.draw_match(img1,img2,cur_corr1[cur_mask_p].cpu().numpy(),cur_corr2[cur_mask_p].cpu().numpy(),inlier=cur_mask_inlier)
        base_name_seq=os.path.join(img_path1.split('/')[-1]+'_'+img_path2.split('/')[-1]+'_'+img_path1.split('/')[-2])
        save_path=os.path.join(config.train_vis_folder,'train_vis',config.log_base,str(step),base_name_seq+'.png')
        cv2.imwrite(save_path,dis_play)