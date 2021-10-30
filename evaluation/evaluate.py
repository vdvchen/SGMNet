import os
from torch.multiprocessing import Process,Manager,set_start_method,Pool
import functools
import argparse
import yaml
import numpy as np
import sys
import cv2
from tqdm import trange
set_start_method('spawn',force=True)


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from components import load_component
from utils import evaluation_utils,metrics

parser = argparse.ArgumentParser(description='dump eval data.')
parser.add_argument('--config_path', type=str, default='configs/eval/scannet_eval_sgm.yaml')
parser.add_argument('--num_process_match', type=int, default=4)
parser.add_argument('--num_process_eval', type=int, default=4)
parser.add_argument('--vis_folder',type=str,default=None)
args=parser.parse_args()    

def feed_match(info,matcher):
    x1,x2,desc1,desc2,size1,size2=info['x1'],info['x2'],info['desc1'],info['desc2'],info['img1'].shape[:2],info['img2'].shape[:2]
    test_data = {'x1': x1,'x2': x2,'desc1': desc1,'desc2': desc2,'size1':np.flip(np.asarray(size1)),'size2':np.flip(np.asarray(size2)) }
    corr1,corr2=matcher.run(test_data)
    return [corr1,corr2]


def reader_handler(config,read_que):
  reader=load_component('reader',config['name'],config)
  for index in range(len(reader)):
    index+=0
    info=reader.run(index)
    read_que.put(info)
  read_que.put('over')


def match_handler(config,read_que,match_que):
  matcher=load_component('matcher',config['name'],config)
  match_func=functools.partial(feed_match,matcher=matcher)
  pool = Pool(args.num_process_match)
  cache=[]
  while True:
    item=read_que.get()
    #clear cache
    if item=='over':
      if len(cache)!=0:
        results=pool.map(match_func,cache)
        for cur_item,cur_result in zip(cache,results):
          cur_item['corr1'],cur_item['corr2']=cur_result[0],cur_result[1]
          match_que.put(cur_item)
      match_que.put('over')
      break
    cache.append(item)
    #print(len(cache))
    if len(cache)==args.num_process_match:
      #matching in parallel
      results=pool.map(match_func,cache)
      for cur_item,cur_result in zip(cache,results):
          cur_item['corr1'],cur_item['corr2']=cur_result[0],cur_result[1]
          match_que.put(cur_item)
      cache=[]
  pool.close()
  pool.join()


def evaluate_handler(config,match_que):
  evaluator=load_component('evaluator',config['name'],config)
  pool = Pool(args.num_process_eval)
  cache=[]
  for _ in trange(config['num_pair']):
    item=match_que.get()
    if item=='over':
      if len(cache)!=0:
        results=pool.map(evaluator.run,cache)
        for cur_res in results:
          evaluator.res_inqueue(cur_res)
      break
    cache.append(item)
    if len(cache)==args.num_process_eval:
      results=pool.map(evaluator.run,cache)
      for cur_res in results:
          evaluator.res_inqueue(cur_res)
      cache=[]
    if args.vis_folder is not None:
      #dump visualization
      corr1_norm,corr2_norm=evaluation_utils.normalize_intrinsic(item['corr1'],item['K1']),\
                            evaluation_utils.normalize_intrinsic(item['corr2'],item['K2'])
      inlier_mask=metrics.compute_epi_inlier(corr1_norm,corr2_norm,item['e'],config['inlier_th'])
      display=evaluation_utils.draw_match(item['img1'],item['img2'],item['corr1'],item['corr2'],inlier_mask)
      cv2.imwrite(os.path.join(args.vis_folder,str(item['index'])+'.png'),display)
  evaluator.parse()


if __name__=='__main__':
  with open(args.config_path, 'r') as f:
    config = yaml.load(f)
  if args.vis_folder is not None and not os.path.exists(args.vis_folder):
    os.mkdir(args.vis_folder)

  read_que,match_que,estimate_que=Manager().Queue(maxsize=100),Manager().Queue(maxsize=100),Manager().Queue(maxsize=100)

  read_process=Process(target=reader_handler,args=(config['reader'],read_que))
  match_process=Process(target=match_handler,args=(config['matcher'],read_que,match_que))
  evaluate_process=Process(target=evaluate_handler,args=(config['evaluator'],match_que))

  read_process.start()
  match_process.start()
  evaluate_process.start()

  read_process.join()
  match_process.join()
  evaluate_process.join()