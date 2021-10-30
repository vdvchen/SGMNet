import os
import glob
import pickle
from tqdm import trange
import numpy as np
import h5py
from numpy.core.fromnumeric import reshape
from .base_dumper import BaseDumper

import sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, ROOT_DIR)
import utils

class fmbench(BaseDumper):
    
    def get_seqs(self):
        data_dir=os.path.join(self.config['rawdata_dir'])
        self.split_list=[]
        for seq in self.config['data_seq']:
            cur_split_list=np.unique(np.loadtxt(os.path.join(data_dir,seq,'pairs_which_dataset.txt'),dtype=str))
            self.split_list.append(cur_split_list)
            for split in cur_split_list:
                split_dir=os.path.join(data_dir,seq,split)
                dump_dir=os.path.join(self.config['feature_dump_dir'],seq,split)
                cur_img_seq=glob.glob(os.path.join(split_dir,'Images','*.jpg'))
                cur_dump_seq=[os.path.join(dump_dir,path.split('/')[-1])+'_'+self.config['extractor']['name']+'_'+str(self.config['extractor']['num_kpt'])\
                             +'.hdf5' for path in cur_img_seq]
                self.img_seq+=cur_img_seq
                self.dump_seq+=cur_dump_seq

    def format_dump_folder(self):
        if not os.path.exists(self.config['feature_dump_dir']):
            os.mkdir(self.config['feature_dump_dir'])
        for seq_index in range(len(self.config['data_seq'])):
            seq_dir=os.path.join(self.config['feature_dump_dir'],self.config['data_seq'][seq_index])
            if not os.path.exists(seq_dir):
                os.mkdir(seq_dir)
            for split in self.split_list[seq_index]:
                split_dir=os.path.join(seq_dir,split)
                if not os.path.exists(split_dir):
                    os.mkdir(split_dir)

    def format_dump_data(self):
        print('Formatting data...')
        self.data={'K1':[],'K2':[],'R':[],'T':[],'e':[],'f':[],'fea_path1':[],'fea_path2':[],'img_path1':[],'img_path2':[]}

        for seq_index in range(len(self.config['data_seq'])):
            seq=self.config['data_seq'][seq_index]
            print(seq)
            pair_list=np.loadtxt(os.path.join(self.config['rawdata_dir'],seq,'pairs_with_gt.txt'),dtype=float)
            which_split_list=np.loadtxt(os.path.join(self.config['rawdata_dir'],seq,'pairs_which_dataset.txt'),dtype=str)

            for pair_index in trange(len(pair_list)):
                cur_pair=pair_list[pair_index]
                cur_split=which_split_list[pair_index]
                index1,index2=int(cur_pair[0]),int(cur_pair[1])
                #get intrinsic
                camera=np.loadtxt(os.path.join(self.config['rawdata_dir'],seq,cur_split,'Camera.txt'),dtype=float)
                K1,K2=camera[index1].reshape([3,3]),camera[index2].reshape([3,3])
                #get pose
                pose=np.loadtxt(os.path.join(self.config['rawdata_dir'],seq,cur_split,'Poses.txt'),dtype=float)
                pose1,pose2=pose[index1].reshape([3,4]),pose[index2].reshape([3,4])
                R1,R2,t1,t2=pose1[:3,:3],pose2[:3,:3],pose1[:3,3][:,np.newaxis],pose2[:3,3][:,np.newaxis]
                dR = np.dot(R2, R1.T)
                dt = t2 - np.dot(dR, t1)
                dt /= np.sqrt(np.sum(dt**2))
                
                e_gt_unnorm = np.reshape(np.matmul(
                np.reshape(utils.evaluation_utils.np_skew_symmetric(dt.astype('float64').reshape(1, 3)), (3, 3)),
                np.reshape(dR.astype('float64'), (3, 3))), (3, 3))
                e_gt = e_gt_unnorm / np.linalg.norm(e_gt_unnorm)

                f=cur_pair[2:].reshape([3,3])
                f_gt=f / np.linalg.norm(f)

                self.data['K1'].append(K1),self.data['K2'].append(K2)
                self.data['R'].append(dR),self.data['T'].append(dt)
                self.data['e'].append(e_gt),self.data['f'].append(f_gt)

                img_path1,img_path2=os.path.join(seq,cur_split,'Images',str(index1).zfill(8)+'.jpg'),\
                                    os.path.join(seq,cur_split,'Images',str(index1).zfill(8)+'.jpg')
                
                fea_path1,fea_path2=os.path.join(self.config['feature_dump_dir'],seq,cur_split,str(index1).zfill(8)+'.jpg'+'_'+self.config['extractor']['name']
                                    +'_'+str(self.config['extractor']['num_kpt'])+'.hdf5'),\
                                    os.path.join(self.config['feature_dump_dir'],seq,cur_split,str(index2).zfill(8)+'.jpg'+'_'+self.config['extractor']['name']
                                    +'_'+str(self.config['extractor']['num_kpt'])+'.hdf5')
                
                self.data['img_path1'].append(img_path1),self.data['img_path2'].append(img_path2)
                self.data['fea_path1'].append(fea_path1),self.data['fea_path2'].append(fea_path2)
            
        self.form_standard_dataset()
