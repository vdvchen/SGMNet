import torch
import torch.optim as optim
from tqdm import trange
import os
from tensorboardX import SummaryWriter
import numpy as np
import cv2
from loss import SGMLoss,SGLoss
from valid import valid,dump_train_vis

import sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)


from utils import train_utils

def train_step(optimizer, model, match_loss, data,step,pre_avg_loss):
    data['step']=step
    result=model(data,test_mode=False)
    loss_res=match_loss.run(data,result)
    
    optimizer.zero_grad()
    loss_res['total_loss'].backward()
    #apply reduce on all record tensor
    for key in loss_res.keys():
        loss_res[key]=train_utils.reduce_tensor(loss_res[key],'mean')
  
    if loss_res['total_loss']<7*pre_avg_loss or step<200 or pre_avg_loss==0:
        optimizer.step()
        unusual_loss=False
    else:
        optimizer.zero_grad()
        unusual_loss=True
    return loss_res,unusual_loss


def train(model, train_loader, valid_loader, config,model_config):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=config.train_lr)
    
    if config.model_name=='SGM':
        match_loss = SGMLoss(config,model_config) 
    elif config.model_name=='SG':
        match_loss= SGLoss(config,model_config)
    else:
        raise NotImplementedError
    
    checkpoint_path = os.path.join(config.log_base, 'checkpoint.pth')
    config.resume = os.path.isfile(checkpoint_path)
    if config.resume:
        if config.local_rank==0:
            print('==> Resuming from checkpoint..')
        checkpoint = torch.load(checkpoint_path,map_location='cuda:{}'.format(config.local_rank))
        model.load_state_dict(checkpoint['state_dict'])
        best_acc = checkpoint['best_acc']
        start_step = checkpoint['step']
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        best_acc = -1
        start_step = 0
    train_loader_iter = iter(train_loader)
    
    if config.local_rank==0:
        writer=SummaryWriter(os.path.join(config.log_base,'log_file'))

    train_loader.sampler.set_epoch(start_step*config.train_batch_size//len(train_loader.dataset))
    pre_avg_loss=0
    
    progress_bar=trange(start_step, config.train_iter,ncols=config.tqdm_width) if config.local_rank==0 else range(start_step, config.train_iter)
    for step in progress_bar:
        try:
            train_data = next(train_loader_iter)
        except StopIteration:
            if config.local_rank==0:
                print('epoch: ',step*config.train_batch_size//len(train_loader.dataset))
            train_loader.sampler.set_epoch(step*config.train_batch_size//len(train_loader.dataset))
            train_loader_iter = iter(train_loader)
            train_data = next(train_loader_iter)
    
        train_data = train_utils.tocuda(train_data)
        lr=min(config.train_lr*config.decay_rate**(step-config.decay_iter),config.train_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # run training
        loss_res,unusual_loss = train_step(optimizer, model, match_loss, train_data,step-start_step,pre_avg_loss)
        if (step-start_step)<=200:
            pre_avg_loss=loss_res['total_loss'].data
        if (step-start_step)>200 and not unusual_loss:
            pre_avg_loss=pre_avg_loss.data*0.9+loss_res['total_loss'].data*0.1
        if unusual_loss and config.local_rank==0:
            print('unusual loss! pre_avg_loss: ',pre_avg_loss,'cur_loss: ',loss_res['total_loss'].data)
        #log
        if config.local_rank==0 and step%config.log_intv==0 and not unusual_loss:
            writer.add_scalar('TotalLoss',loss_res['total_loss'],step)
            writer.add_scalar('CorrLoss',loss_res['loss_corr'],step)
            writer.add_scalar('InCorrLoss', loss_res['loss_incorr'], step)
            writer.add_scalar('dustbin', model.module.dustbin, step)

            if config.model_name=='SGM':
                writer.add_scalar('SeedConfLoss', loss_res['loss_seed_conf'], step)
                writer.add_scalar('MidCorrLoss', loss_res['loss_corr_mid'].sum(), step)
                writer.add_scalar('MidInCorrLoss', loss_res['loss_incorr_mid'].sum(), step)
            

        # valid ans save
        b_save = ((step + 1) % config.save_intv) == 0
        b_validate = ((step + 1) % config.val_intv) == 0
        if b_validate:
            total_loss,acc_corr,acc_incorr,seed_precision_tower,seed_recall_tower,acc_mid=valid(valid_loader, model, match_loss, config,model_config)
            if config.local_rank==0:
                writer.add_scalar('ValidAcc', acc_corr, step)
                writer.add_scalar('ValidLoss', total_loss, step)
                
                if config.model_name=='SGM':
                    for i in range(len(seed_recall_tower)):
                        writer.add_scalar('seed_conf_pre_%d'%i,seed_precision_tower[i],step)
                        writer.add_scalar('seed_conf_recall_%d' % i, seed_precision_tower[i], step)
                    for i in range(len(acc_mid)):
                        writer.add_scalar('acc_mid%d'%i,acc_mid[i],step)
                    print('acc_corr: ',acc_corr.data,'acc_incorr: ',acc_incorr.data,'seed_conf_pre: ',seed_precision_tower.mean().data,
                     'seed_conf_recall: ',seed_recall_tower.mean().data,'acc_mid: ',acc_mid.mean().data)
                else:
                     print('acc_corr: ',acc_corr.data,'acc_incorr: ',acc_incorr.data)
                
                #saving best
                if acc_corr > best_acc:
                    print("Saving best model with va_res = {}".format(acc_corr))
                    best_acc = acc_corr
                    save_dict={'step': step + 1,
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer' : optimizer.state_dict()}
                    save_dict.update(save_dict)
                    torch.save(save_dict, os.path.join(config.log_base, 'model_best.pth'))

        if b_save:
            if config.local_rank==0:
                save_dict={'step': step + 1,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict()}
                torch.save(save_dict, checkpoint_path)
            
            #draw match results
            model.eval()
            with torch.no_grad():
                if config.local_rank==0:
                    if not os.path.exists(os.path.join(config.train_vis_folder,'train_vis')):
                        os.mkdir(os.path.join(config.train_vis_folder,'train_vis'))
                    if not os.path.exists(os.path.join(config.train_vis_folder,'train_vis',config.log_base)):
                        os.mkdir(os.path.join(config.train_vis_folder,'train_vis',config.log_base))
                    os.mkdir(os.path.join(config.train_vis_folder,'train_vis',config.log_base,str(step)))
                res=model(train_data)
                dump_train_vis(res,train_data,step,config)
            model.train()
    
    if config.local_rank==0:
        writer.close()
