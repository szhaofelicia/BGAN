import sys
import argparse
import gc
import logging
import os
import numpy as np
import torch

from sgan.data.loader import data_loader
from sgan.utils import int_tuple, bool_flag, get_total_norm

data_dir='/media/felicia/Data/basketball-partial'
# data_dir='/media/felicia/Data/nba2016'


parser = argparse.ArgumentParser()
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

# Dataset options
parser.add_argument('--dataset_name', default='01.04.2016.TOR.at.CLE-partial', type=str)  # default:zara1
parser.add_argument('--delim', default='\t')  # default: ' '
parser.add_argument('--loader_num_workers', default=4, type=int)
parser.add_argument('--obs_len', default=8, type=int)
parser.add_argument('--pred_len', default=8, type=int)
parser.add_argument('--skip', default=1, type=int)
parser.add_argument('--metric', default='meter', type=str) # original dataset: foot, default: foot to meter

# Optimization
parser.add_argument('--batch_size', default=8, type=int)  # 32
parser.add_argument('--num_iterations', default=20000, type=int)  # default:10000
parser.add_argument('--num_epochs', default=500, type=int)

# Model Options
parser.add_argument('--embedding_dim', default=16, type=int)  # 64
parser.add_argument('--num_layers', default=1, type=int)
parser.add_argument('--dropout', default=0, type=float)
parser.add_argument('--batch_norm', default=0, type=bool_flag)  # default:0-bool_flag
parser.add_argument('--mlp_dim', default=64, type=int)  # default: 1024

# Generator Options
parser.add_argument('--encoder_h_dim_g', default=32, type=int)  # default:64
parser.add_argument('--decoder_h_dim_g', default=32, type=int)  # default:128
parser.add_argument('--noise_dim', default=(8,), type=int_tuple)  # default: None-int_tuple
parser.add_argument('--noise_type', default='gaussian')
parser.add_argument('--noise_mix_type', default='global')  # default:pred
parser.add_argument('--clipping_threshold_g', default=1.5, type=float)  # default:0
parser.add_argument('--g_learning_rate', default=1e-3, type=float)  # default:5e-4,0.001
parser.add_argument('--g_steps', default=1, type=int)

# Discriminator Options
parser.add_argument('--d_type', default='local', type=str)  # default:'local'
parser.add_argument('--encoder_h_dim_d', default=64, type=int)  # default:64
parser.add_argument('--d_learning_rate', default=1e-3, type=float)  # default:5e-4, 0.001
parser.add_argument('--d_steps', default=2, type=int)  # default:2
parser.add_argument('--clipping_threshold_d', default=0, type=float)
parser.add_argument('--d_activation', default='relu', type=str)  # 'relu'

# Pooling Options
parser.add_argument('--pooling_type', default='pool_net')  # default:'pool_net'
parser.add_argument('--pool_every_timestep', default=0, type=bool_flag)  # default:1-bool_flag

# Pool Net Option
parser.add_argument('--bottleneck_dim', default=32, type=int)  # 1024

# Social Pooling Options
parser.add_argument('--neighborhood_size', default=2.0, type=float)
parser.add_argument('--grid_size', default=8, type=int)

# Loss Options
parser.add_argument('--l2_loss_weight', default=1, type=float)  # default:0->1
parser.add_argument('--best_k', default=10, type=int)  # default:1
parser.add_argument('--l2_loss_mode', default="raw", type=str)  # default:"raw"

# Output
# parser.add_argument('--output_dir', default=output_dir)  # os.getcwd()
parser.add_argument('--print_every', default=10, type=int)  # default:5
parser.add_argument('--checkpoint_every', default=50, type=int)  # default:100
parser.add_argument('--checkpoint_name', default='basketball_den_gsw')
parser.add_argument('--checkpoint_start_from', default=None)
parser.add_argument('--restore_from_checkpoint', default=0, type=int)  # default:1
parser.add_argument('--num_samples_check', default=5000, type=int)

# Misc
parser.add_argument('--use_gpu', default=1, type=int)  # 1: use_gpu
parser.add_argument('--timing', default=0, type=int)
parser.add_argument('--gpu_num', default="0", type=str)

args = parser.parse_args()

tmp_path= os.path.join(data_dir,'01.02.2016.DEN.at.GSW','tmp') # 200 files:0-199
# tmp_path= os.path.join(data_dir,args.dataset_name,'train_sample') #
tmp_dset, tmp_loader = data_loader(args, tmp_path)


dataset_len=len(tmp_dset)
print(dataset_len)
iterations_per_epoch = dataset_len / 128 / args.d_steps
if args.num_epochs:
    args.num_iterations = int(iterations_per_epoch * args.num_epochs)

print(iterations_per_epoch)
print(args.num_iterations)

# traj_max=[]

# for batch in tmp_loader:
#     # batch = [tensor.cuda() for tensor in batch]
#     (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
#      obs_team_vec, obs_pos_vec,non_linear_ped, loss_mask, seq_start_end) = batch
#     # print(obs_traj.shape) # (obs_len, batch, 32
#     # print(obs_team_vec.shape) # (obs_len, batch, 3)
#     # print(obs_pos_vec.shape) # (obs_len, batch, 4)
#     # print(obs_traj.shape)
#
#     # tmp=torch.max(torch.flatten(obs_traj))
#     # traj_max.append(tmp)
#
# # xy_max=max(traj_max)
# # print(xy_max)