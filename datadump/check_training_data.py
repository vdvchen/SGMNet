import argparse
import os
import numpy as np
import h5py
import cv2
from numpy.core.numeric import indices
import pyxis as px
from tqdm import trange

import sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from utils import evaluation_utils,train_utils

parser = argparse.ArgumentParser(description='checking training data.')
parser.add_argument('--meta_dir', type=str, default='dataset/valid')
parser.add_argument('--dataset_dir', type=str, default='dataset')
parser.add_argument('--desc_dir', type=str, default='desc')
parser.add_argument('--raw_dir', type=str, default='raw_data')
parser.add_argument('--desc_suffix', type=str, default='_root_1000.hdf5')
parser.add_argument('--vis_folder',type=str,default=None)
args=parser.parse_args()  



if __name__=='__main__':
    if args.vis_folder is not None and not os.path.exists(args.vis_folder):
        os.mkdir(args.vis_folder)

    pair_num_list=np.loadtxt(os.path.join(args.meta_dir,'pair_num.txt'),dtype=str)
    pair_seq_list,accu_pair_list=train_utils.parse_pair_seq(pair_num_list)
    total_pair=int(pair_num_list[0,1])
    total_inlier_rate,total_corr_num,total_incorr_num=[],[],[]
    pair_num_list=pair_num_list[1:]

    for index in trange(total_pair):
        seq=pair_seq_list[index]
        index_within_seq=index-accu_pair_list[seq]
        with h5py.File(os.path.join(args.dataset_dir,seq,'info.h5py'),'r') as data:
            corr=data['corr'][str(index_within_seq)][()]
            corr1,corr2=corr[:,0],corr[:,1]
            incorr1,incorr2=data['incorr1'][str(index_within_seq)][()],data['incorr2'][str(index_within_seq)][()]
            img_path1,img_path2=data['img_path1'][str(index_within_seq)][()][0].decode(),data['img_path2'][str(index_within_seq)][()][0].decode()
            img_name1,img_name2=img_path1.split('/')[-1],img_path2.split('/')[-1]
            fea_path1,fea_path2=os.path.join(args.desc_dir,seq,img_name1+args.desc_suffix),os.path.join(args.desc_dir,seq,img_name2+args.desc_suffix)
            with h5py.File(fea_path1,'r') as fea1, h5py.File(fea_path2,'r') as fea2:
                desc1,kpt1=fea1['descriptors'][()],fea1['keypoints'][()][:,:2]
                desc2,kpt2=fea2['descriptors'][()],fea2['keypoints'][()][:,:2]
            sim_mat=desc1@desc2.T
            nn_index1,nn_index2=np.argmax(sim_mat,axis=1),np.argmax(sim_mat,axis=0)
            mask_mutual=(nn_index2[nn_index1]==np.arange(len(nn_index1)))[corr1]
            mask_inlier=nn_index1[corr1]==corr2
            mask_nn_correct=np.logical_and(mask_mutual,mask_inlier)
            #statistics
            total_inlier_rate.append(mask_nn_correct.mean())
            total_corr_num.append(len(corr1))
            total_incorr_num.append((len(incorr1)+len(incorr2))/2)
            #dump visualization
            if args.vis_folder is not None:
                #draw corr
                img1,img2=cv2.imread(os.path.join(args.raw_dir,img_path1)),cv2.imread(os.path.join(args.raw_dir,img_path2))
                corr1_pos,corr2_pos=np.take_along_axis(kpt1,corr1[:,np.newaxis],axis=0),np.take_along_axis(kpt2,corr2[:,np.newaxis],axis=0)
                dis_corr=evaluation_utils.draw_match(img1,img2,corr1_pos,corr2_pos)
                cv2.imwrite(os.path.join(args.vis_folder,str(index)+'.png'),dis_corr)
                #draw incorr
                incorr1_pos,incorr2_pos=np.take_along_axis(kpt1,incorr1[:,np.newaxis],axis=0),np.take_along_axis(kpt2,incorr2[:,np.newaxis],axis=0)
                dis_incorr1,dis_incorr2=evaluation_utils.draw_points(img1,incorr1_pos),evaluation_utils.draw_points(img2,incorr2_pos)
                cv2.imwrite(os.path.join(args.vis_folder,str(index)+'_incorr1.png'),dis_incorr1)
                cv2.imwrite(os.path.join(args.vis_folder,str(index)+'_incorr2.png'),dis_incorr2)

    print('NN matching accuracy: ',np.asarray(total_inlier_rate).mean())
    print('mean corr number: ',np.asarray(total_corr_num).mean())
    print('mean incorr number: ',np.asarray(total_incorr_num).mean())
