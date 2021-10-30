OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='0' python -m torch.distributed.launch --nproc_per_node=1 --master_port 23003 main.py \
--model_name=SG \
--config_path=configs/sg.yaml \
--rawdata_path=rawdata \
--desc_path=desc_path \
--desc_suffix=_root_1000.hdf5 \
--dataset_path=dataset_path  \
--log_base=log_root_1k_sg \
--num_kpt=1000 \
--train_iter=900000