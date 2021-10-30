import torch
import yaml
import time
from collections import OrderedDict,namedtuple
import os
import sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from sgmnet import matcher as SGM_Model
from superglue import matcher as SG_Model


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--matcher_name', type=str, default='SGM',
  help='number of processes.')
parser.add_argument('--config_path', type=str, default='configs/cost/sgm_cost.yaml',
  help='number of processes.')
parser.add_argument('--num_kpt', type=int, default=4000,
  help='keypoint number, default:100')
parser.add_argument('--iter_num', type=int, default=100,
  help='keypoint number, default:100')


def test_cost(test_data,model):
    with torch.no_grad():
        #warm up call
        _=model(test_data)
        torch.cuda.synchronize()
        a=time.time()
        for _ in range(int(args.iter_num)):
            _=model(test_data)
        torch.cuda.synchronize()
        b=time.time()
    print('Average time per run(ms): ',(b-a)/args.iter_num*1e3)
    print('Peak memory(MB): ',torch.cuda.max_memory_allocated()/1e6)


if __name__=='__main__':
    torch.backends.cudnn.benchmark=False
    args = parser.parse_args()
    with open(args.config_path, 'r') as f:
      model_config = yaml.load(f)
    model_config=namedtuple('model_config',model_config.keys())(*model_config.values())
    
    if args.matcher_name=='SGM':
      model = SGM_Model(model_config) 
    elif args.matcher_name=='SG':
      model = SG_Model(model_config)
    model.cuda(),model.eval()
    
    test_data = {
            'x1':torch.rand(1,args.num_kpt,2).cuda()-0.5,
            'x2':torch.rand(1,args.num_kpt,2).cuda()-0.5,
            'desc1': torch.rand(1,args.num_kpt,128).cuda(),
            'desc2': torch.rand(1,args.num_kpt,128).cuda()
            }

    test_cost(test_data,model)
