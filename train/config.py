import argparse

def str2bool(v):
    return v.lower() in ("true", "1")


arg_lists = []
parser = argparse.ArgumentParser()


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


# -----------------------------------------------------------------------------
# Network
net_arg = add_argument_group("Network")
net_arg.add_argument(
    "--model_name", type=str,default='SGM', help=""
    "model for training")
net_arg.add_argument(
    "--config_path", type=str,default='configs/sgm.yaml', help=""
    "config path for model")

# -----------------------------------------------------------------------------
# Data
data_arg = add_argument_group("Data")
data_arg.add_argument(
    "--rawdata_path", type=str, default='rawdata', help=""
    "path for rawdata")
data_arg.add_argument(
    "--dataset_path", type=str, default='dataset', help=""
    "path for dataset")
data_arg.add_argument(
    "--desc_path", type=str, default='desc', help=""
    "path for descriptor(kpt) dir")
data_arg.add_argument(
    "--num_kpt", type=int, default=1000, help=""
    "number of kpt for training")
data_arg.add_argument(
    "--input_normalize", type=str, default='img', help=""
    "normalize type for input kpt, img or intrinsic")
data_arg.add_argument(
    "--data_aug", type=str2bool, default=True, help=""
    "apply kpt coordinate homography augmentation")
data_arg.add_argument(
    "--desc_suffix", type=str, default='suffix', help=""
    "desc file suffix")


# -----------------------------------------------------------------------------
# Loss
loss_arg = add_argument_group("loss")
loss_arg.add_argument(
    "--momentum", type=float, default=0.9, help=""
    "momentum")
loss_arg.add_argument(
    "--seed_loss_weight", type=float, default=250, help=""
    "confidence loss weight for sgm")
loss_arg.add_argument(
    "--mid_loss_weight", type=float, default=1, help=""
    "midseeding loss weight for sgm")
loss_arg.add_argument(
    "--inlier_th", type=float, default=5e-3, help=""
    "inlier threshold for epipolar distance (for sgm and visualization)")


# -----------------------------------------------------------------------------
# Training
train_arg = add_argument_group("Train")
train_arg.add_argument(
    "--train_lr", type=float, default=1e-4, help=""
    "learning rate")
train_arg.add_argument(
    "--train_batch_size", type=int, default=16, help=""
    "batch size")
train_arg.add_argument(
    "--gpu_id", type=str,default='0', help='id(s) for CUDA_VISIBLE_DEVICES')
train_arg.add_argument(
    "--train_iter", type=int, default=1000000, help=""
    "training iterations to perform")
train_arg.add_argument(
    "--log_base", type=str, default="./log/", help=""
    "log path")
train_arg.add_argument(
    "--val_intv", type=int, default=20000, help=""
    "validation interval")
train_arg.add_argument(
    "--save_intv", type=int, default=1000, help=""
    "summary interval")
train_arg.add_argument(
    "--log_intv", type=int, default=100, help=""
    "log interval")
train_arg.add_argument(
    "--decay_rate", type=float, default=0.999996, help=""
    "lr decay rate")
train_arg.add_argument(
    "--decay_iter", type=float, default=300000, help=""
    "lr decay iter")
train_arg.add_argument(
    "--local_rank", type=int, default=0, help=""
    "local rank for ddp")
train_arg.add_argument(
    "--train_vis_folder", type=str, default='.', help=""
    "visualization folder during training")

# -----------------------------------------------------------------------------
# Visualization
vis_arg = add_argument_group('Visualization')
vis_arg.add_argument(
    "--tqdm_width", type=int, default=79, help=""
    "width of the tqdm bar"
)

def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed


def print_usage():
    parser.print_usage()

#
# config.py ends here