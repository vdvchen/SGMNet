import numpy as np
import torch
import torch.utils.data as data
import cv2
import os
import h5py
import random

import sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, ROOT_DIR)

from utils import train_utils,evaluation_utils

torch.multiprocessing.set_sharing_strategy('file_system')


class Offline_Dataset(data.Dataset):
    def __init__(self,config,mode):
        assert mode=='train' or mode=='valid'

        self.config = config
        self.mode = mode
        metadir=os.path.join(config.dataset_path,'valid') if mode=='valid' else os.path.join(config.dataset_path,'train')
        
        pair_num_list=np.loadtxt(os.path.join(metadir,'pair_num.txt'),dtype=str)
        self.total_pairs=int(pair_num_list[0,1])
        self.pair_seq_list,self.accu_pair_num=train_utils.parse_pair_seq(pair_num_list)


    def collate_fn(self, batch):
        batch_size, num_pts = len(batch), batch[0]['x1'].shape[0]
        
        data = {}
        dtype=['x1','x2','kpt1','kpt2','desc1','desc2','num_corr','num_incorr1','num_incorr2','e_gt','pscore1','pscore2','img_path1','img_path2']
        for key in dtype:
            data[key]=[]
        for sample in batch:
            for key in dtype:
                data[key].append(sample[key])
           
        for key in ['x1', 'x2','kpt1','kpt2', 'desc1', 'desc2','e_gt','pscore1','pscore2']:
            data[key] = torch.from_numpy(np.stack(data[key])).float()
        for key in ['num_corr', 'num_incorr1', 'num_incorr2']:
            data[key] = torch.from_numpy(np.stack(data[key])).int()

        # kpt augmentation with random homography
        if (self.mode == 'train' and self.config.data_aug):
            homo_mat = torch.from_numpy(train_utils.get_rnd_homography(batch_size)).unsqueeze(1)
            aug_seed=random.random() 
            if aug_seed<0.5:
                x1_homo = torch.cat([data['x1'], torch.ones([batch_size, num_pts, 1])], dim=-1).unsqueeze(-1)
                x1_homo = torch.matmul(homo_mat.float(), x1_homo.float()).squeeze(-1)
                data['aug_x1'] = x1_homo[:, :, :2] / x1_homo[:, :, 2].unsqueeze(-1)
                data['aug_x2']=data['x2']
            else:
                x2_homo = torch.cat([data['x2'], torch.ones([batch_size, num_pts, 1])], dim=-1).unsqueeze(-1)
                x2_homo = torch.matmul(homo_mat.float(), x2_homo.float()).squeeze(-1)
                data['aug_x2'] = x2_homo[:, :, :2] / x2_homo[:, :, 2].unsqueeze(-1)
                data['aug_x1']=data['x1']
        else:
            data['aug_x1'],data['aug_x2']=data['x1'],data['x2']
        return data


    def __getitem__(self, index):
        seq=self.pair_seq_list[index]
        index_within_seq=index-self.accu_pair_num[seq]

        with h5py.File(os.path.join(self.config.dataset_path,seq,'info.h5py'),'r') as data:
            R,t = data['dR'][str(index_within_seq)][()], data['dt'][str(index_within_seq)][()]
            egt = np.reshape(np.matmul(np.reshape(evaluation_utils.np_skew_symmetric(t.astype('float64').reshape(1, 3)), (3, 3)),np.reshape(R.astype('float64'), (3, 3))), (3, 3))
            egt = egt / np.linalg.norm(egt)
            K1, K2 = data['K1'][str(index_within_seq)][()],data['K2'][str(index_within_seq)][()]
            size1,size2=data['size1'][str(index_within_seq)][()],data['size2'][str(index_within_seq)][()]

            img_path1,img_path2=data['img_path1'][str(index_within_seq)][()][0].decode(),data['img_path2'][str(index_within_seq)][()][0].decode()
            img_name1,img_name2=img_path1.split('/')[-1],img_path2.split('/')[-1]
            img_path1,img_path2=os.path.join(self.config.rawdata_path,img_path1),os.path.join(self.config.rawdata_path,img_path2)
            fea_path1,fea_path2=os.path.join(self.config.desc_path,seq,img_name1+self.config.desc_suffix),\
                                os.path.join(self.config.desc_path,seq,img_name2+self.config.desc_suffix)
            with h5py.File(fea_path1,'r') as fea1, h5py.File(fea_path2,'r') as fea2:
                desc1,kpt1,pscore1=fea1['descriptors'][()],fea1['keypoints'][()][:,:2],fea1['keypoints'][()][:,2]
                desc2,kpt2,pscore2=fea2['descriptors'][()],fea2['keypoints'][()][:,:2],fea2['keypoints'][()][:,2]
                kpt1,kpt2,desc1,desc2=kpt1[:self.config.num_kpt],kpt2[:self.config.num_kpt],desc1[:self.config.num_kpt],desc2[:self.config.num_kpt]

            # normalize kpt
            if self.config.input_normalize=='intrinsic':
                x1, x2 = np.concatenate([kpt1, np.ones([kpt1.shape[0], 1])], axis=-1), np.concatenate(
                    [kpt2, np.ones([kpt2.shape[0], 1])], axis=-1)
                x1, x2 = np.matmul(np.linalg.inv(K1), x1.T).T[:, :2], np.matmul(np.linalg.inv(K2), x2.T).T[:, :2]
            elif self.config.input_normalize=='img' :   
                x1,x2=(kpt1-size1/2)/size1,(kpt2-size2/2)/size2
                S1_inv,S2_inv=np.asarray([[size1[0],0,0.5*size1[0]],[0,size1[1],0.5*size1[1]],[0,0,1]]),\
                            np.asarray([[size2[0],0,0.5*size2[0]],[0,size2[1],0.5*size2[1]],[0,0,1]])
                M1,M2=np.matmul(np.linalg.inv(K1),S1_inv),np.matmul(np.linalg.inv(K2),S2_inv)
                egt=np.matmul(np.matmul(M2.transpose(),egt),M1)
                egt = egt / np.linalg.norm(egt)
            else:
                raise NotImplementedError

            corr=data['corr'][str(index_within_seq)][()]
            incorr1,incorr2=data['incorr1'][str(index_within_seq)][()],data['incorr2'][str(index_within_seq)][()]
            
        #permute kpt
        valid_corr=corr[corr.max(axis=-1)<self.config.num_kpt]
        valid_incorr1,valid_incorr2=incorr1[incorr1<self.config.num_kpt],incorr2[incorr2<self.config.num_kpt]
        num_corr, num_incorr1, num_incorr2 = len(valid_corr), len(valid_incorr1), len(valid_incorr2)
        mask1_invlaid, mask2_invalid = np.ones(x1.shape[0]).astype(bool), np.ones(x2.shape[0]).astype(bool)
        mask1_invlaid[valid_corr[:, 0]] = False
        mask2_invalid[valid_corr[:, 1]] = False
        mask1_invlaid[valid_incorr1] = False
        mask2_invalid[valid_incorr2] = False
        invalid_index1,invalid_index2=np.nonzero(mask1_invlaid)[0],np.nonzero(mask2_invalid)[0]

        #random sample from point w/o valid annotation 
        cur_kpt1 = self.config.num_kpt - num_corr - num_incorr1
        cur_kpt2 = self.config.num_kpt - num_corr - num_incorr2

        if (invalid_index1.shape[0] < cur_kpt1):
            sub_idx1 = np.concatenate([np.arange(len(invalid_index1)),np.random.randint(len(invalid_index1),size=cur_kpt1-len(invalid_index1))])
        if (invalid_index1.shape[0] >= cur_kpt1):
            sub_idx1 =np.random.choice(len(invalid_index1), cur_kpt1,replace=False)
        if (invalid_index2.shape[0] < cur_kpt2):
            sub_idx2 = np.concatenate([np.arange(len(invalid_index2)),np.random.randint(len(invalid_index2),size=cur_kpt2-len(invalid_index2))])
        if (invalid_index2.shape[0] >= cur_kpt2):
            sub_idx2 = np.random.choice(len(invalid_index2), cur_kpt2,replace=False)
        
        per_idx1,per_idx2=np.concatenate([valid_corr[:,0],valid_incorr1,invalid_index1[sub_idx1]]),\
                          np.concatenate([valid_corr[:,1],valid_incorr2,invalid_index2[sub_idx2]])
        
        pscore1,pscore2=pscore1[per_idx1][:,np.newaxis],pscore2[per_idx2][:,np.newaxis]
        x1,x2=x1[per_idx1][:,:2],x2[per_idx2][:,:2]
        desc1,desc2=desc1[per_idx1],desc2[per_idx2]
        kpt1,kpt2=kpt1[per_idx1],kpt2[per_idx2]
        
        return {'x1': x1, 'x2': x2, 'kpt1':kpt1,'kpt2':kpt2,'desc1': desc1, 'desc2': desc2, 'num_corr': num_corr, 'num_incorr1': num_incorr1,'num_incorr2': num_incorr2,'e_gt':egt,\
                'pscore1':pscore1,'pscore2':pscore2,'img_path1':img_path1,'img_path2':img_path2}

    def __len__(self):
        return self.total_pairs


