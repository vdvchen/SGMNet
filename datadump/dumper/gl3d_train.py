import os
import glob
import math
import re
import numpy as np
import h5py
from tqdm import trange
from torch.multiprocessing import Pool
import pyxis as px
from .base_dumper import BaseDumper

import sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, ROOT_DIR)

from utils import transformations,data_utils

class gl3d_train(BaseDumper):
    
    def get_seqs(self):
        data_dir=os.path.join(self.config['rawdata_dir'],'data')
        seq_train=np.loadtxt(os.path.join(self.config['rawdata_dir'],'list','comb','imageset_train.txt'),dtype=str)
        seq_valid=np.loadtxt(os.path.join(self.config['rawdata_dir'],'list','comb','imageset_test.txt'),dtype=str)

        #filtering seq list
        self.seq_list,self.train_list,self.valid_list=[],[],[]
        for seq in seq_train:
            if seq not in self.config['exclude_seq']:
                self.train_list.append(seq)
        for seq in seq_valid:
            if seq not in self.config['exclude_seq']:
                self.valid_list.append(seq)
        seq_list=[]
        if self.config['dump_train']:
            seq_list.append(self.train_list)
        if self.config['dump_valid']:
            seq_list.append(self.valid_list)
        self.seq_list=np.concatenate(seq_list,axis=0)

        #self.seq_list=self.seq_list[:2]
        #self.valid_list=self.valid_list[:2]
        for seq in self.seq_list:
            dump_dir=os.path.join(self.config['feature_dump_dir'],seq)
            cur_img_seq=glob.glob(os.path.join(data_dir,seq,'undist_images','*.jpg'))
            cur_dump_seq=[os.path.join(dump_dir,path.split('/')[-1])+'_'+self.config['extractor']['name']+'_'+str(self.config['extractor']['num_kpt'])\
                             +'.hdf5' for path in cur_img_seq]
            self.img_seq+=cur_img_seq
            self.dump_seq+=cur_dump_seq


    def format_dump_folder(self):
        if not os.path.exists(self.config['feature_dump_dir']):
            os.mkdir(self.config['feature_dump_dir'])
        for seq in self.seq_list:
            seq_dir=os.path.join(self.config['feature_dump_dir'],seq)
            if not os.path.exists(seq_dir):
                os.mkdir(seq_dir)
        if not os.path.exists(self.config['dataset_dump_dir']):
            os.mkdir(self.config['dataset_dump_dir'])
    

    def load_geom(self,seq):
        # load geometry file
        geom_file=os.path.join(self.config['rawdata_dir'],'data',seq,'geolabel','cameras.txt')
        basename_list=np.loadtxt(os.path.join(self.config['rawdata_dir'],'data',seq,'basenames.txt'),dtype=str)
        geom_dict = []
        cameras = np.loadtxt(geom_file)
        camera_index=0
        for base_index in range(len(basename_list)):
            if base_index<cameras[camera_index][0]:
                geom_dict.append(None)
                continue
            cur_geom = {}
            ori_img_size = [cameras[camera_index][-2], cameras[camera_index][-1]]
            scale_factor = [1000. / ori_img_size[0], 1000. / ori_img_size[1]]
            K = np.asarray([[cameras[camera_index][1], cameras[camera_index][5], cameras[camera_index][3]],
                            [0, cameras[camera_index][2], cameras[camera_index][4]],
                            [0, 0, 1]])
            # Rescale calbration according to previous resizing
            S = np.asarray([[scale_factor[0], 0, 0],
                            [0, scale_factor[1], 0],
                            [0, 0, 1]])
            K = np.dot(S, K)
            cur_geom["K"] = K
            cur_geom['R'] = cameras[camera_index][9:18].reshape([3, 3])
            cur_geom['T'] = cameras[camera_index][6:9]
            cur_geom['size']=np.asarray([1000,1000])
            geom_dict.append(cur_geom)
            camera_index+=1
        return geom_dict


    def load_depth(self,file_path):
        with open(os.path.join(file_path), 'rb') as fin:
            color = None
            width = None
            height = None
            scale = None
            data_type = None
            header = str(fin.readline().decode('UTF-8')).rstrip()
            if header == 'PF':
                color = True
            elif header == 'Pf':
                color = False
            else:
                raise Exception('Not a PFM file.')
            dim_match = re.match(r'^(\d+)\s(\d+)\s$', fin.readline().decode('UTF-8'))
            if dim_match:
                width, height = map(int, dim_match.groups())
            else:
                raise Exception('Malformed PFM header.')
            scale = float((fin.readline().decode('UTF-8')).rstrip())
            if scale < 0:  # little-endian
                data_type = '<f'
            else:
                data_type = '>f'  # big-endian
            data_string = fin.read()
            data = np.fromstring(data_string, data_type)
            shape = (height, width, 3) if color else (height, width)
            data = np.reshape(data, shape)
            data = np.flip(data, 0)
        return data


    def dump_info(self,seq,info):
        pair_type=['dR','dt','K1','K2','size1','size2','corr','incorr1','incorr2']
        num_pairs=len(info['dR'])
        os.mkdir(os.path.join(self.config['dataset_dump_dir'],seq))
        with h5py.File(os.path.join(self.config['dataset_dump_dir'],seq,'info.h5py'), 'w') as f:
            for type in pair_type:
                dg=f.create_group(type)
                for idx in range(num_pairs):
                    data_item=np.asarray(info[type][idx])
                    dg.create_dataset(str(idx),data_item.shape,data_item.dtype,data=data_item)
            for type in ['img_path1','img_path2']:
                dg=f.create_group(type)
                for idx in range(num_pairs):
                    dg.create_dataset(str(idx),[1],h5py.string_dtype(encoding='ascii'),data=info[type][idx].encode('ascii'))
        
        with open(os.path.join(self.config['dataset_dump_dir'],seq,'pair_num.txt'), 'w') as f:
            f.write(str(info['pair_num']))

    def format_seq(self,index):
        seq=self.seq_list[index]
        seq_dir=os.path.join(os.path.join(self.config['rawdata_dir'],'data',seq))
        basename_list=np.loadtxt(os.path.join(seq_dir,'basenames.txt'),dtype=str)
        pair_list=np.loadtxt(os.path.join(seq_dir,'geolabel','common_track.txt'),dtype=float)[:,:2].astype(int)
        overlap_score=np.loadtxt(os.path.join(seq_dir,'geolabel','common_track.txt'),dtype=float)[:,2]
        geom_dict=self.load_geom(seq)

        #check info existance
        if os.path.exists(os.path.join(self.config['dataset_dump_dir'],seq,'pair_num.txt')):
            return

        angle_list=[]
        #filtering pairs
        for cur_pair in pair_list:
            pair_index1,pair_index2=cur_pair[0],cur_pair[1]
            geo1,geo2=geom_dict[pair_index1],geom_dict[pair_index2]
            dR = np.dot(geo2['R'], geo1['R'].T)
            q = transformations.quaternion_from_matrix(dR)
            angle_list.append(math.acos(q[0]) * 2 * 180 / math.pi)
        angle_list=np.asarray(angle_list)
        mask_survive=np.logical_and(
                            np.logical_and(angle_list>self.config['angle_th'][0],angle_list<self.config['angle_th'][1]),
                            np.logical_and(overlap_score>self.config['overlap_th'][0],overlap_score<self.config['overlap_th'][1])
                        )
        pair_list=pair_list[mask_survive]
        if len(pair_list)<100:
            print(seq,len(pair_list))
        #sample pairs
        shuffled_pair_list=np.random.permutation(pair_list)
        sample_target=min(self.config['pairs_per_seq'],len(shuffled_pair_list))
        sample_number=0

        info={'dR':[],'dt':[],'K1':[],'K2':[],'img_path1':[],'img_path2':[],'fea_path1':[],'fea_path2':[],'size1':[],'size2':[],
            'corr':[],'incorr1':[],'incorr2':[],'pair_num':[]}
        for cur_pair in shuffled_pair_list:
            pair_index1,pair_index2=cur_pair[0],cur_pair[1]
            geo1,geo2=geom_dict[pair_index1],geom_dict[pair_index2]
            dR = np.dot(geo2['R'], geo1['R'].T)
            t1, t2 = geo1["T"].reshape([3, 1]), geo2["T"].reshape([3, 1])
            dt = t2 - np.dot(dR, t1)
            K1,K2=geo1['K'],geo2['K']
            size1,size2=geo1['size'],geo2['size']

            basename1,basename2=basename_list[pair_index1],basename_list[pair_index2]
            img_path1,img_path2=os.path.join(seq,'undist_images',basename1+'.jpg'),os.path.join(seq,'undist_images',basename2+'.jpg')
            fea_path1,fea_path2=os.path.join(seq,basename1+'.jpg'+'_'+self.config['extractor']['name']+'_'+str(self.config['extractor']['num_kpt'])+'.hdf5'),\
                                os.path.join(seq,basename2+'.jpg'+'_'+self.config['extractor']['name']+'_'+str(self.config['extractor']['num_kpt'])+'.hdf5')

            with h5py.File(os.path.join(self.config['feature_dump_dir'],fea_path1),'r') as fea1, \
                h5py.File(os.path.join(self.config['feature_dump_dir'],fea_path2),'r') as fea2:
                desc1,desc2=fea1['descriptors'][()],fea2['descriptors'][()]
                kpt1,kpt2=fea1['keypoints'][()],fea2['keypoints'][()]
                depth_path1,depth_path2=os.path.join(self.config['rawdata_dir'],'data',seq,'depths',basename1+'.pfm'),\
                                        os.path.join(self.config['rawdata_dir'],'data',seq,'depths',basename2+'.pfm')
                depth1,depth2=self.load_depth(depth_path1),self.load_depth(depth_path2)
                corr_index,incorr_index1,incorr_index2=data_utils.make_corr(kpt1[:,:2],kpt2[:,:2],desc1,desc2,depth1,depth2,K1,K2,dR,dt,size1,size2,
                                                                            self.config['corr_th'],self.config['incorr_th'],self.config['check_desc'])
            
            if len(corr_index)>self.config['min_corr'] and len(incorr_index1)>self.config['min_incorr'] and len(incorr_index2)>self.config['min_incorr']:
                info['corr'].append(corr_index),info['incorr1'].append(incorr_index1),info['incorr2'].append(incorr_index2)
                info['dR'].append(dR),info['dt'].append(dt),info['K1'].append(K1),info['K2'].append(K2),info['img_path1'].append(img_path1),info['img_path2'].append(img_path2)
                info['fea_path1'].append(fea_path1),info['fea_path2'].append(fea_path2),info['size1'].append(size1),info['size2'].append(size2)
                sample_number+=1
            if sample_number==sample_target:
                break
        info['pair_num']=sample_number
        #dump info
        self.dump_info(seq,info)

  
    def collect_meta(self):
        print('collecting meta info...')
        dump_path,seq_list=[],[]
        if self.config['dump_train']:
            dump_path.append(os.path.join(self.config['dataset_dump_dir'],'train'))
            seq_list.append(self.train_list)
        if self.config['dump_valid']:
            dump_path.append(os.path.join(self.config['dataset_dump_dir'],'valid'))
            seq_list.append(self.valid_list)
        for pth,seqs in zip(dump_path,seq_list):
            if not os.path.exists(pth):
                os.mkdir(pth)
            pair_num_list,total_pair=[],0
            for seq_index in range(len(seqs)):    
                seq=seqs[seq_index]
                pair_num=np.loadtxt(os.path.join(self.config['dataset_dump_dir'],seq,'pair_num.txt'),dtype=int)
                pair_num_list.append(str(pair_num))
                total_pair+=pair_num
            pair_num_list=np.stack([np.asarray(seqs,dtype=str),np.asarray(pair_num_list,dtype=str)],axis=1)
            pair_num_list=np.concatenate([np.asarray([['total',str(total_pair)]]),pair_num_list],axis=0)
            np.savetxt(os.path.join(pth,'pair_num.txt'),pair_num_list,fmt='%s')
            
    def format_dump_data(self):
        print('Formatting data...')
        iteration_num=len(self.seq_list)//self.config['num_process']
        if len(self.seq_list)%self.config['num_process']!=0:
            iteration_num+=1
        pool=Pool(self.config['num_process'])
        for index in trange(iteration_num):
            indices=range(index*self.config['num_process'],min((index+1)*self.config['num_process'],len(self.seq_list)))
            pool.map(self.format_seq,indices)
        pool.close()
        pool.join()

        self.collect_meta()