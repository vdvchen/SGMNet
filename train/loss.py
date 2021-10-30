import torch
import numpy as np


def batch_episym(x1, x2, F):
    batch_size, num_pts = x1.shape[0], x1.shape[1]
    x1 = torch.cat([x1, x1.new_ones(batch_size, num_pts,1)], dim=-1).reshape(batch_size, num_pts,3,1)
    x2 = torch.cat([x2, x2.new_ones(batch_size, num_pts,1)], dim=-1).reshape(batch_size, num_pts,3,1)
    F = F.reshape(-1,1,3,3).repeat(1,num_pts,1,1)
    x2Fx1 = torch.matmul(x2.transpose(2,3), torch.matmul(F, x1)).reshape(batch_size,num_pts)
    Fx1 = torch.matmul(F,x1).reshape(batch_size,num_pts,3)
    Ftx2 = torch.matmul(F.transpose(2,3),x2).reshape(batch_size,num_pts,3)
    ys = (x2Fx1**2 * (
            1.0 / (Fx1[:, :, 0]**2 + Fx1[:, :, 1]**2 + 1e-15) +
            1.0 / (Ftx2[:, :, 0]**2 + Ftx2[:, :, 1]**2 + 1e-15))).sqrt()
    return ys
    

def CELoss(seed_x1,seed_x2,e,confidence,inlier_th,batch_mask=1):
    #seed_x: b*k*2
    ys=batch_episym(seed_x1,seed_x2,e)
    mask_pos,mask_neg=(ys<=inlier_th).float(),(ys>inlier_th).float()
    num_pos,num_neg=torch.relu(torch.sum(mask_pos, dim=1) - 1.0) + 1.0,torch.relu(torch.sum(mask_neg, dim=1) - 1.0) + 1.0
    loss_pos,loss_neg=-torch.log(abs(confidence) + 1e-8)*mask_pos,-torch.log(abs(1-confidence)+1e-8)*mask_neg
    classif_loss = torch.mean(loss_pos * 0.5 / num_pos.unsqueeze(-1) + loss_neg * 0.5 / num_neg.unsqueeze(-1),dim=-1)
    classif_loss =classif_loss*batch_mask
    classif_loss=classif_loss.mean()
    precision = torch.mean(
        torch.sum((confidence > 0.5).type(confidence.type()) * mask_pos, dim=1) /
        (torch.sum((confidence > 0.5).type(confidence.type()), dim=1)+1e-8)
    )
    recall = torch.mean(
        torch.sum((confidence > 0.5).type(confidence.type()) * mask_pos, dim=1) /
        num_pos
    )
    return classif_loss,precision,recall


def CorrLoss(desc_mat,batch_num_corr,batch_num_incorr1,batch_num_incorr2):
    total_loss_corr,total_loss_incorr=0,0
    total_acc_corr,total_acc_incorr=0,0
    batch_size = desc_mat.shape[0]
    log_p=torch.log(abs(desc_mat)+1e-8)

    for i in range(batch_size):
        cur_log_p=log_p[i]
        num_corr=batch_num_corr[i]
        num_incorr1,num_incorr2=batch_num_incorr1[i],batch_num_incorr2[i]
     
        #loss and acc
        loss_corr = -torch.diag(cur_log_p)[:num_corr].mean()
        loss_incorr=(-cur_log_p[num_corr:num_corr+num_incorr1,-1].mean()-cur_log_p[-1,num_corr:num_corr+num_incorr2].mean())/2

        value_row, row_index = torch.max(desc_mat[i,:-1,:-1], dim=-1)
        value_col, col_index = torch.max(desc_mat[i,:-1,:-1], dim=-2)
        acc_incorr=((value_row[num_corr:num_corr+num_incorr1]<0.2).float().mean()+
                    (value_col[num_corr:num_corr+num_incorr2]<0.2).float().mean())/2

        acc_row_mask = row_index[:num_corr] == torch.arange(num_corr).cuda()
        acc_col_mask = col_index[:num_corr] == torch.arange(num_corr).cuda()
        acc = (acc_col_mask & acc_row_mask).float().mean()
     
        total_loss_corr+=loss_corr
        total_loss_incorr+=loss_incorr
        total_acc_corr += acc
        total_acc_incorr+=acc_incorr

    total_acc_corr/=batch_size
    total_acc_incorr/=batch_size
    total_loss_corr/=batch_size
    total_loss_incorr/=batch_size
    return total_loss_corr,total_loss_incorr,total_acc_corr,total_acc_incorr


class SGMLoss:
    def __init__(self,config,model_config):
        self.config=config
        self.model_config=model_config

    def run(self,data,result):
        loss_corr,loss_incorr,acc_corr,acc_incorr=CorrLoss(result['p'],data['num_corr'],data['num_incorr1'],data['num_incorr2'])
        loss_mid_corr_tower,loss_mid_incorr_tower,acc_mid_tower=[],[],[]
        
        #mid loss
        for i in range(len(result['mid_p'])):
            mid_p=result['mid_p'][i]
            loss_mid_corr,loss_mid_incorr,mid_acc_corr,mid_acc_incorr=CorrLoss(mid_p,data['num_corr'],data['num_incorr1'],data['num_incorr2'])
            loss_mid_corr_tower.append(loss_mid_corr),loss_mid_incorr_tower.append(loss_mid_incorr),acc_mid_tower.append(mid_acc_corr)
        if len(result['mid_p']) != 0:
            loss_mid_corr_tower,loss_mid_incorr_tower, acc_mid_tower = torch.stack(loss_mid_corr_tower), torch.stack(loss_mid_incorr_tower), torch.stack(acc_mid_tower)
        else:
            loss_mid_corr_tower,loss_mid_incorr_tower, acc_mid_tower= torch.zeros(1).cuda(), torch.zeros(1).cuda(),torch.zeros(1).cuda()
  
        #seed confidence loss
        classif_loss_tower,classif_precision_tower,classif_recall_tower=[],[],[]
        for layer in range(len(result['seed_conf'])):
            confidence=result['seed_conf'][layer]
            seed_index=result['seed_index'][(np.asarray(self.model_config.seedlayer)<=layer).nonzero()[0][-1]]
            seed_x1,seed_x2=data['x1'].gather(dim=1, index=seed_index[:,:,0,None].expand(-1, -1,2)),\
                            data['x2'].gather(dim=1, index=seed_index[:,:,1,None].expand(-1, -1,2))
            classif_loss,classif_precision,classif_recall=CELoss(seed_x1,seed_x2,data['e_gt'],confidence,self.config.inlier_th)
            classif_loss_tower.append(classif_loss), classif_precision_tower.append(classif_precision), classif_recall_tower.append(classif_recall)
        classif_loss, classif_precision_tower, classif_recall_tower=torch.stack(classif_loss_tower).mean(),torch.stack(classif_precision_tower), \
                                                                    torch.stack(classif_recall_tower)
       
            
        classif_loss*=self.config.seed_loss_weight
        loss_mid_corr_tower*=self.config.mid_loss_weight
        loss_mid_incorr_tower*=self.config.mid_loss_weight
        total_loss=loss_corr+loss_incorr+classif_loss+loss_mid_corr_tower.sum()+loss_mid_incorr_tower.sum()

        return {'loss_corr':loss_corr,'loss_incorr':loss_incorr,'acc_corr':acc_corr,'acc_incorr':acc_incorr,'loss_seed_conf':classif_loss,
                'pre_seed_conf':classif_precision_tower,'recall_seed_conf':classif_recall_tower,'loss_corr_mid':loss_mid_corr_tower,
                'loss_incorr_mid':loss_mid_incorr_tower,'mid_acc_corr':acc_mid_tower,'total_loss':total_loss}
        
class SGLoss:
    def __init__(self,config,model_config):
        self.config=config
        self.model_config=model_config
        
    def run(self,data,result):
        loss_corr,loss_incorr,acc_corr,acc_incorr=CorrLoss(result['p'],data['num_corr'],data['num_incorr1'],data['num_incorr2'])
        total_loss=loss_corr+loss_incorr
        return {'loss_corr':loss_corr,'loss_incorr':loss_incorr,'acc_corr':acc_corr,'acc_incorr':acc_incorr,'total_loss':total_loss}
     