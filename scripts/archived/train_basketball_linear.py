import sys
import argparse
import gc
import logging
import os

sys.path.append("/scratch/sz2257/sgan")
sys.path.append("../../")
import time
import json
# import yaml

from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

from sgan.data.loader import data_loader
from sgan.losses import  l2_loss
from sgan.losses import displacement_error, final_displacement_error

from sgan.models_linear import TrajectoryLinearRegressor

from sgan.utils import int_tuple, bool_flag, get_total_norm
from sgan.utils import relative_to_abs, get_dset_path


torch.backends.cudnn.benchmark = True
writer = SummaryWriter()

time_str="_".join(writer.get_logdir().split("/")[1].split("_")[:2])

# output_dir="/media/felicia/Data/sgan_results/{}".format(time_str)
output_dir="/scratch/sz2257/sgan_results/{}".format(time_str)

# data_dir='/media/felicia/Data/basketball-partial'
data_dir='/scratch/sz2257/basketball-partial'

parser = argparse.ArgumentParser()
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

# Dataset options
parser.add_argument('--dataset_name', default='01.04.2016.TOR.at.CLE-partial', type=str) #default:zara1
parser.add_argument('--dataset_dir', default=data_dir, type=str)
parser.add_argument('--delim', default='\t') #default: ' '
parser.add_argument('--loader_num_workers', default=4, type=int)
parser.add_argument('--obs_len', default=8, type=int)
parser.add_argument('--pred_len', default=8, type=int)
parser.add_argument('--skip', default=1, type=int)
parser.add_argument('--metric', default="meter", type=str)
parser.add_argument("--model", default="team_pos", type=str)

# Optimization
parser.add_argument('--batch_size', default=128, type=int) #32
parser.add_argument('--num_iterations', default=20000, type=int) #default:10000
parser.add_argument('--num_epochs', default=500, type=int)

# Model Options
parser.add_argument('--embedding_dim', default=16, type=int) #64
parser.add_argument('--num_layers', default=1, type=int)
parser.add_argument('--dropout', default=0, type=float)
parser.add_argument('--tp_dropout', default=0, type=float)
parser.add_argument('--batch_norm', default=0, type=bool_flag) #default:0-bool_flag
parser.add_argument('--mlp_dim', default=64, type=int) #default: 1024
parser.add_argument('--team_embedding_dim', default=16, type=int) #default: 1024
parser.add_argument('--pos_embedding_dim', default=32, type=int) #default: 1024
parser.add_argument('--interaction_activation', default="none", type=str)

# Generator Options
parser.add_argument('--encoder_h_dim_g', default=32, type=int) #default:64
parser.add_argument('--decoder_h_dim_g', default=32, type=int) #default:128
parser.add_argument('--noise_dim', default=(8,), type=int_tuple) # default: None-int_tuple
parser.add_argument('--noise_type', default='gaussian')
parser.add_argument('--noise_mix_type', default='global') #default:pred
parser.add_argument('--clipping_threshold_g', default=1.5, type=float) #default:0
parser.add_argument('--g_learning_rate', default=1e-3, type=float) #default:5e-4,0.001
parser.add_argument('--g_steps', default=1, type=int)

# Discriminator Options
parser.add_argument('--d_type', default='local', type=str) #default:'local'
parser.add_argument('--encoder_h_dim_d', default=64, type=int) #default:64
parser.add_argument('--d_learning_rate', default=1e-3, type=float) #default:5e-4, 0.001
parser.add_argument('--d_steps', default=2, type=int) #default:2
parser.add_argument('--clipping_threshold_d', default=0, type=float)
parser.add_argument('--d_activation', default='relu', type=str) # 'relu'


# Pooling Options
parser.add_argument('--pooling_type', default='pool_net') #default:'pool_net'
parser.add_argument('--pool_every_timestep', default=0, type=bool_flag) #default:1-bool_flag

# Pool Net Option
parser.add_argument('--bottleneck_dim', default=32, type=int) # 1024

# Social Pooling Options
parser.add_argument('--neighborhood_size', default=2.0, type=float)
parser.add_argument('--grid_size', default=8, type=int)

# Loss Options
parser.add_argument('--l2_loss_weight', default=1, type=float) #default:0->1
parser.add_argument('--best_k', default=10, type=int) #default:1
parser.add_argument('--l2_loss_mode', default="raw", type=str) #default:"raw"


# Output
parser.add_argument('--output_dir', default=output_dir) # os.getcwd()
parser.add_argument('--print_every', default=10, type=int) #default:5
parser.add_argument('--checkpoint_every', default=50, type=int) #default:100
parser.add_argument('--checkpoint_name', default='basketball_linear')
parser.add_argument('--checkpoint_start_from', default=None)
parser.add_argument('--restore_from_checkpoint', default=0, type=int) #default:1
parser.add_argument('--num_samples_check', default=5000, type=int)
parser.add_argument("--tb_path", default=writer.get_logdir(), type=str)

# Misc
parser.add_argument('--use_gpu', default=1, type=int) # 1: use_gpu
parser.add_argument('--timing', default=0, type=int)
parser.add_argument('--gpu_num', default="0", type=str)

def init_weights(m):
    classname = m.__class__.__name__
    # if classname.find('Linear') != -1:
    if type(m)==nn.Linear:
        nn.init.kaiming_normal_(m.weight)


def get_dtypes(args):
    long_dtype = torch.LongTensor
    float_dtype = torch.FloatTensor
    if args.use_gpu == 1:
        long_dtype = torch.cuda.LongTensor
        float_dtype = torch.cuda.FloatTensor
    return long_dtype, float_dtype

# TrajectoryDiscriminator = None
# TrajectoryGenerator = None
def main(args):
    print(args)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    # train_path = get_dset_path(args.dataset_name, 'train')
    # val_path = get_dset_path(args.dataset_name, 'val')
    train_path= os.path.join(args.dataset_dir,args.dataset_name,'train_sample') # 10 files:0-9
    val_path= os.path.join(args.dataset_dir,args.dataset_name,'val_sample') # 5 files: 10-14

    long_dtype, float_dtype = get_dtypes(args)

    logger.info("Initializing train dataset")
    train_dset, train_loader = data_loader(args, train_path)
    logger.info("Initializing val dataset")
    _, val_loader = data_loader(args, val_path)

    iterations_per_epoch = len(train_dset) / args.batch_size / args.d_steps
    if args.num_epochs:
        args.num_iterations = int(iterations_per_epoch * args.num_epochs)

    logger.info(
        'There are {} iterations per epoch'.format(iterations_per_epoch)
    )

    regressor = TrajectoryLinearRegressor(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=args.embedding_dim,
        mlp_dim=args.mlp_dim,
        dropout=args.dropout,
        batch_norm=args.batch_norm,
    )

    regressor.apply(init_weights)
    regressor.type(float_dtype).train()
    logger.info('Here is the regressor:')
    logger.info(regressor)


    optimizer_r = optim.Adam(regressor.parameters(), lr=args.g_learning_rate)


    # Maybe restore from checkpoint
    restore_path = None
    if args.checkpoint_start_from is not None:
        restore_path = args.checkpoint_start_from
    elif args.restore_from_checkpoint == 1:
        restore_path = os.path.join(args.output_dir,
                                    '%s_with_model.pt' % args.checkpoint_name)

    if restore_path is not None and os.path.isfile(restore_path):
        logger.info('Restoring from checkpoint {}'.format(restore_path))
        checkpoint = torch.load(restore_path)
        regressor.load_state_dict(checkpoint['r_state'])
        optimizer_r.load_state_dict(checkpoint['r_optim_state'])
        t = checkpoint['counters']['t']
        epoch = checkpoint['counters']['epoch']
        checkpoint['restore_ts'].append(t)
    else:
        # Starting from scratch, so initialize checkpoint data structure
        t, epoch = 0, 0
        checkpoint = {
            'args': args.__dict__,
            'R_losses': defaultdict(list),
            'losses_ts': [],
            'metrics_val': defaultdict(list),
            'metrics_train': defaultdict(list),
            'sample_ts': [],
            'restore_ts': [],
            'norm_r': [],
            'counters': {
                't': None,
                'epoch': None,
            },
            'r_state': None,
            'r_optim_state': None,
            'r_best_state': None,
            'best_t': None,
            'g_best_nl_state': None,
            'd_best_state_nl': None,
            'best_t_nl': None,
        }
    t0 = None
    while t < args.num_iterations:
        gc.collect()
        epoch += 1
        logger.info('Starting epoch {}'.format(epoch))
        for batch in train_loader:
            if args.timing == 1:
                torch.cuda.synchronize()
                t1 = time.time()

            # Decide whether to use the batch for stepping on discriminator or
            # generator; an iteration consists of args.d_steps steps on the
            # discriminator followed by args.g_steps steps on the generator.


            step_type = 'r'
            losses_r = regressor_step(args, batch, regressor,optimizer_r)
            checkpoint['norm_r'].append(
                get_total_norm(regressor.parameters())
            )

            if args.timing == 1:
                torch.cuda.synchronize()
                t2 = time.time()
                logger.info('{} step took {}'.format(step_type, t2 - t1))

            if args.timing == 1:
                if t0 is not None:
                    logger.info('Interation {} took {}'.format(
                        t - 1, time.time() - t0
                    ))
                t0 = time.time()

            # Maybe save loss
            if t % args.print_every == 0:
                logger.info('t = {} / {}'.format(t + 1, args.num_iterations))
                for k, v in sorted(losses_r.items()):
                    logger.info('  [R] {}: {:.3f}'.format(k, v))
                    checkpoint['R_losses'][k].append(v)
                checkpoint['losses_ts'].append(t)

                ## log scalars
                for k, v in sorted(losses_r.items()):
                    writer.add_scalar("loss/{}".format(k), v, t)

            # Maybe save a checkpoint
            if t > 0 and t % args.checkpoint_every == 0:
                checkpoint['counters']['t'] = t
                checkpoint['counters']['epoch'] = epoch
                checkpoint['sample_ts'].append(t)

                # Check stats on the validation set
                logger.info('Checking stats on val ...')
                metrics_val = check_accuracy(
                    args, val_loader, regressor
                )
                logger.info('Checking stats on train ...')
                metrics_train = check_accuracy(
                    args, train_loader, regressor,limit=True
                )

                for k, v in sorted(metrics_val.items()):
                    logger.info('  [val] {}: {:.3f}'.format(k, v))
                    checkpoint['metrics_val'][k].append(v)
                for k, v in sorted(metrics_train.items()):
                    logger.info('  [train] {}: {:.3f}'.format(k, v))
                    checkpoint['metrics_train'][k].append(v)

                ## log scalars
                for k, v in sorted(metrics_val.items()):
                    writer.add_scalar("val/{}".format(k), v, t)
                for k, v in sorted(metrics_train.items()):
                    writer.add_scalar("train/{}".format(k), v, t)

                min_ade = min(checkpoint['metrics_val']['ade'])
                min_ade_nl = min(checkpoint['metrics_val']['ade_nl'])

                if metrics_val['ade'] == min_ade:
                    logger.info('New low for avg_disp_error')
                    checkpoint['best_t'] = t
                    checkpoint['r_best_state'] = regressor.state_dict()

                if metrics_val['ade_nl'] == min_ade_nl:
                    logger.info('New low for avg_disp_error_nl')
                    checkpoint['best_t_nl'] = t
                    checkpoint['r_best_nl_state'] = regressor.state_dict()

                # Save another checkpoint with model weights and
                # optimizer state
                checkpoint['r_state'] = regressor.state_dict()
                checkpoint['r_optim_state'] = regressor.state_dict()

                # checkpoint_path = os.path.join(
                #     args.output_dir, '{}_with_model_{:06d}.pt'.format(args.checkpoint_name,t)
                # )
                checkpoint_path = os.path.join(args.output_dir, '{}_with_model.pt'.format(args.checkpoint_name))
                logger.info('Saving checkpoint to {}'.format(checkpoint_path))
                torch.save(checkpoint, checkpoint_path)
                logger.info('Done.')

                # Save a checkpoint with no model weights by making a shallow
                # copy of the checkpoint excluding some items

                # checkpoint_path = os.path.join(
                #     args.output_dir, '{}_no_model_{:06d}.pt' .format(args.checkpoint_name,t))

                checkpoint_path = os.path.join(args.output_dir, '{}_no_model.pt' .format(args.checkpoint_name))
                logger.info('Saving checkpoint to {}'.format(checkpoint_path))
                key_blacklist = [
                    'r_state',  'r_best_state', 'r_best_nl_state','r_optim_state',
                ]
                small_checkpoint = {}
                for k, v in checkpoint.items():
                    if k not in key_blacklist:
                        small_checkpoint[k] = v
                torch.save(small_checkpoint, checkpoint_path)
                logger.info('Done.')

            t += 1
            if t >= args.num_iterations:
                break


        ## log scalars
        # for k, v in sorted(losses_d.items()):
        #     writer.add_scalar("train/{}".format(k),v,epoch)
        # for k, v in sorted(losses_g.items()):
        #     writer.add_scalar("train/{}".format(k),v,epoch)

def regressor_step(
    args, batch, regressor, optimizer_r
):
    batch = [tensor.cuda() for tensor in batch]
    (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
     obs_team_vec, obs_pos_vec, pred_team_vec, pred_pos_vec,
     non_linear_ped, loss_mask, seq_start_end) = batch
    losses = {}
    loss = torch.zeros(1).to(pred_traj_gt)
    r_l2_loss_rel = []

    loss_mask = loss_mask[:, args.obs_len:]

    for _ in range(args.best_k):
        generator_out = regressor(obs_traj, obs_traj_rel, seq_start_end)


        pred_traj_fake_rel = generator_out

        if args.l2_loss_weight > 0:
            r_l2_loss_rel.append(args.l2_loss_weight * l2_loss(
                pred_traj_fake_rel,
                pred_traj_gt_rel,
                loss_mask,
                mode=args.l2_loss_mode # default:"raw"
            ))

    r_l2_loss_sum_rel = torch.zeros(1).to(pred_traj_gt)
    if args.l2_loss_weight > 0:
        r_l2_loss_rel = torch.stack(r_l2_loss_rel, dim=1)
        for start, end in seq_start_end.data:
            _r_l2_loss_rel = r_l2_loss_rel[start:end]
            _r_l2_loss_rel = torch.sum(_r_l2_loss_rel, dim=0)
            _r_l2_loss_rel = torch.min(_r_l2_loss_rel) / torch.sum(
                loss_mask[start:end])
            r_l2_loss_sum_rel += _r_l2_loss_rel
        losses['R_l2_loss_rel'] = r_l2_loss_sum_rel.item()
        loss += r_l2_loss_sum_rel

    losses['R_total_loss'] = loss.item()

    optimizer_r.zero_grad()
    loss.backward()
    if args.clipping_threshold_g > 0:
        nn.utils.clip_grad_norm_(
            regressor.parameters(), args.clipping_threshold_g
        )
    optimizer_r.step()

    return losses


def check_accuracy(
    args, loader, regressor, limit=False
):
    metrics = {}
    r_l2_losses_abs, r_l2_losses_rel = ([],) * 2
    disp_error, disp_error_l, disp_error_nl = ([],) * 3
    f_disp_error, f_disp_error_l, f_disp_error_nl = ([],) * 3
    total_traj, total_traj_l, total_traj_nl = 0, 0, 0
    loss_mask_sum = 0
    regressor.eval()
    with torch.no_grad():
        for batch in loader:
            batch = [tensor.cuda() for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
             obs_team_vec, obs_pos_vec, pred_team_vec, pred_pos_vec,
             non_linear_ped, loss_mask, seq_start_end) = batch
            linear_ped = 1 - non_linear_ped
            loss_mask = loss_mask[:, args.obs_len:]

            pred_traj_fake_rel = regressor(
                obs_traj, obs_traj_rel, seq_start_end
            )
            pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

            r_l2_loss_abs, r_l2_loss_rel = cal_l2_losses(
                pred_traj_gt, pred_traj_gt_rel, pred_traj_fake,
                pred_traj_fake_rel, loss_mask
            )
            ade, ade_l, ade_nl = cal_ade(
                pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped
            )

            fde, fde_l, fde_nl = cal_fde(
                pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped
            )


            r_l2_losses_abs.append(r_l2_loss_abs.item())
            r_l2_losses_rel.append(r_l2_loss_rel.item())
            disp_error.append(ade.item())
            disp_error_l.append(ade_l.item())
            disp_error_nl.append(ade_nl.item())
            f_disp_error.append(fde.item())
            f_disp_error_l.append(fde_l.item())
            f_disp_error_nl.append(fde_nl.item())

            loss_mask_sum += torch.numel(loss_mask.data)
            total_traj += pred_traj_gt.size(1)
            total_traj_l += torch.sum(linear_ped).item()
            total_traj_nl += torch.sum(non_linear_ped).item()
            if limit and total_traj >= args.num_samples_check:
                break

    metrics['r_l2_loss_abs'] = sum(r_l2_losses_abs) / loss_mask_sum
    metrics['r_l2_loss_rel'] = sum(r_l2_losses_rel) / loss_mask_sum

    metrics['ade'] = sum(disp_error) / (total_traj * args.pred_len)
    metrics['fde'] = sum(f_disp_error) / total_traj
    if total_traj_l != 0:
        metrics['ade_l'] = sum(disp_error_l) / (total_traj_l * args.pred_len)
        metrics['fde_l'] = sum(f_disp_error_l) / total_traj_l
    else:
        metrics['ade_l'] = 0
        metrics['fde_l'] = 0
    if total_traj_nl != 0:
        metrics['ade_nl'] = sum(disp_error_nl) / (
            total_traj_nl * args.pred_len)
        metrics['fde_nl'] = sum(f_disp_error_nl) / total_traj_nl
    else:
        metrics['ade_nl'] = 0
        metrics['fde_nl'] = 0

    regressor.train()
    return metrics


def cal_l2_losses(
    pred_traj_gt, pred_traj_gt_rel, pred_traj_fake, pred_traj_fake_rel,
    loss_mask
):
    g_l2_loss_abs = l2_loss(
        pred_traj_fake, pred_traj_gt, loss_mask, mode='sum'
    )
    g_l2_loss_rel = l2_loss(
        pred_traj_fake_rel, pred_traj_gt_rel, loss_mask, mode='sum'
    )
    return g_l2_loss_abs, g_l2_loss_rel


def cal_ade(pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped):
    ade = displacement_error(pred_traj_fake, pred_traj_gt)
    ade_l = displacement_error(pred_traj_fake, pred_traj_gt, linear_ped)
    ade_nl = displacement_error(pred_traj_fake, pred_traj_gt, non_linear_ped)
    return ade, ade_l, ade_nl


def cal_fde(
    pred_traj_gt, pred_traj_fake, linear_ped, non_linear_ped
):
    fde = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1])
    fde_l = final_displacement_error(
        pred_traj_fake[-1], pred_traj_gt[-1], linear_ped
    )
    fde_nl = final_displacement_error(
        pred_traj_fake[-1], pred_traj_gt[-1], non_linear_ped
    )
    return fde, fde_l, fde_nl


if __name__ == '__main__':
    args = parser.parse_args()

    # TrajectoryGenerator, TrajectoryDiscriminator = MODELS[args.model]
    log_path="{}/config.txt".format(writer.get_logdir())
    with open(log_path,"a") as f:
        json.dump(args.__dict__,f,indent=2)

    # log_path="{}/config.yaml".format(writer.get_logdir())
    # with open(log_path,'w') as file:
    #     args_file=yaml.dump(args,file)
    # print(args_file)
    writer = SummaryWriter(args.tb_path)
    main(args)
    writer.flush()


