import torch.utils.data
from dataset import Offline_Dataset
import yaml
from sgmnet.match_model import matcher as SGM_Model
from superglue.match_model import matcher as SG_Model
import torch.distributed as dist
import torch
import os
from collections import namedtuple
from train import train
from config import get_config, print_usage


def main(config,model_config):
    """The main function."""
    # Initialize network
    if config.model_name=='SGM':
        model = SGM_Model(model_config)
    elif config.model_name=='SG':
        model= SG_Model(model_config)
    else:
        raise NotImplementedError

    #initialize ddp
    torch.cuda.set_device(config.local_rank)
    device = torch.device(f'cuda:{config.local_rank}')
    model.to(device)
    dist.init_process_group(backend='nccl',init_method='env://')
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.local_rank])
    
    if config.local_rank==0:
        os.system('nvidia-smi')

    #initialize dataset
    train_dataset = Offline_Dataset(config,'train')
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,shuffle=True)
    train_loader=torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size//torch.distributed.get_world_size(),
            num_workers=8//dist.get_world_size(), pin_memory=False,sampler=train_sampler,collate_fn=train_dataset.collate_fn)
    
    valid_dataset = Offline_Dataset(config,'valid')
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset,shuffle=False)
    valid_loader=torch.utils.data.DataLoader(valid_dataset, batch_size=config.train_batch_size,
                num_workers=8//dist.get_world_size(), pin_memory=False,collate_fn=valid_dataset.collate_fn,sampler=valid_sampler)
    
    if config.local_rank==0:
        print('start training .....')
    train(model,train_loader, valid_loader, config,model_config)

if __name__ == "__main__":
    # ----------------------------------------
    # Parse configuration
    config, unparsed = get_config()
    with open(config.config_path, 'r') as f:
        model_config = yaml.load(f)
    model_config=namedtuple('model_config',model_config.keys())(*model_config.values())
    # If we have unparsed arguments, print usage and exit
    if len(unparsed) > 0:
        print_usage()
        exit(1)

    main(config,model_config)
