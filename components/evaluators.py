import numpy as np
import sys
import os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from utils import evaluation_utils,metrics,fm_utils
import cv2

class auc_eval:
    def __init__(self,config):
        self.config=config
        self.err_r,self.err_t,self.err=[],[],[]
        self.ms=[]
        self.precision=[]

    def run(self,info):
        E,r_gt,t_gt=info['e'],info['r_gt'],info['t_gt']
        K1,K2,img1,img2=info['K1'],info['K2'],info['img1'],info['img2']
        corr1,corr2=info['corr1'],info['corr2']
        corr1,corr2=evaluation_utils.normalize_intrinsic(corr1,K1),evaluation_utils.normalize_intrinsic(corr2,K2)
        size1,size2=max(img1.shape),max(img2.shape)
        scale1,scale2=self.config['rescale']/size1,self.config['rescale']/size2
        #ransac
        ransac_th=4./((K1[0,0]+K1[1,1])*scale1+(K2[0,0]+K2[1,1])*scale2)
        R_hat,t_hat,E_hat=self.estimate(corr1,corr2,ransac_th)
        #get pose error
        err_r, err_t=metrics.evaluate_R_t(r_gt,t_gt,R_hat,t_hat)
        err=max(err_r,err_t)
        
        if len(corr1)>1:
            inlier_mask=metrics.compute_epi_inlier(corr1,corr2,E,self.config['inlier_th'])
            precision=inlier_mask.mean()
            ms=inlier_mask.sum()/len(info['x1'])
        else:
            ms=precision=0
        
        return {'err_r':err_r,'err_t':err_t,'err':err,'ms':ms,'precision':precision}

    def res_inqueue(self,res):
        self.err_r.append(res['err_r']),self.err_t.append(res['err_t']),self.err.append(res['err'])
        self.ms.append(res['ms']),self.precision.append(res['precision'])

    def estimate(self,corr1,corr2,th):
        num_inlier = -1
        if corr1.shape[0] >= 5:
            E, mask_new = cv2.findEssentialMat(corr1, corr2,method=cv2.RANSAC, threshold=th,prob=1-1e-5)
            if E is None:
                E=[np.eye(3)]
            for _E in np.split(E, len(E) / 3):
                _num_inlier, _R, _t, _ = cv2.recoverPose(_E, corr1, corr2,np.eye(3), 1e9,mask=mask_new)
                if _num_inlier > num_inlier:
                    num_inlier = _num_inlier
                    R = _R
                    t = _t
                    E = _E
        else:
            E,R,t=np.eye(3),np.eye(3),np.zeros(3)
        return R,t,E

    def parse(self):
        ths = np.arange(7) * 5
        approx_auc=metrics.approx_pose_auc(self.err,ths)
        exact_auc=metrics.pose_auc(self.err,ths)
        mean_pre,mean_ms=np.mean(np.asarray(self.precision)),np.mean(np.asarray(self.ms))
        
        print('auc th: ',ths[1:])
        print('approx auc: ',approx_auc)
        print('exact auc: ', exact_auc)
        print('mean match score: ',mean_ms*100)
        print('mean precision: ',mean_pre*100)

        

class FMbench_eval:

    def __init__(self,config):
        self.config=config
        self.pre,self.pre_post,self.sgd=[],[],[]
        self.num_corr,self.num_corr_post=[],[]

    def run(self,info):
        corr1,corr2=info['corr1'],info['corr2']
        F=info['f']
        img1,img2=info['img1'],info['img2']

        if len(corr1)>1:
            pre_bf=fm_utils.compute_inlier_rate(corr1,corr2,np.flip(img1.shape[:2]),np.flip(img2.shape[:2]),F,th=self.config['inlier_th']).mean()
            F_hat,mask_F=cv2.findFundamentalMat(corr1,corr2,method=cv2.FM_RANSAC,ransacReprojThreshold=1,confidence=1-1e-5)
            if F_hat is None:
                F_hat=np.ones([3,3])
                mask_F=np.ones([len(corr1)]).astype(bool)
            else:
                mask_F=mask_F.squeeze().astype(bool)
            F_hat=F_hat[:3]
            pre_af=fm_utils.compute_inlier_rate(corr1[mask_F],corr2[mask_F],np.flip(img1.shape[:2]),np.flip(img2.shape[:2]),F,th=self.config['inlier_th']).mean()
            num_corr_af=mask_F.sum()
            num_corr=len(corr1)
            sgd=fm_utils.compute_SGD(F,F_hat,np.flip(img1.shape[:2]),np.flip(img2.shape[:2]))
        else:
            pre_bf,pre_af,sgd=0,0,1e8
            num_corr,num_corr_af=0,0
        return {'pre':pre_bf,'pre_post':pre_af,'sgd':sgd,'num_corr':num_corr,'num_corr_post':num_corr_af}


    def res_inqueue(self,res):
        self.pre.append(res['pre']),self.pre_post.append(res['pre_post']),self.sgd.append(res['sgd'])
        self.num_corr.append(res['num_corr']),self.num_corr_post.append(res['num_corr_post'])

    def parse(self):
        for seq_index in range(len(self.config['seq'])):
            seq=self.config['seq'][seq_index]
            offset=seq_index*1000
            pre=np.asarray(self.pre)[offset:offset+1000].mean()
            pre_post=np.asarray(self.pre_post)[offset:offset+1000].mean()
            num_corr=np.asarray(self.num_corr)[offset:offset+1000].mean()
            num_corr_post=np.asarray(self.num_corr_post)[offset:offset+1000].mean()
            f_recall=(np.asarray(self.sgd)[offset:offset+1000]<self.config['sgd_inlier_th']).mean()

            print(seq,'results:')
            print('F_recall: ',f_recall)
            print('precision: ',pre)
            print('precision_post: ',pre_post)
            print('num_corr: ',num_corr)
            print('num_corr_post: ',num_corr_post,'\n')


