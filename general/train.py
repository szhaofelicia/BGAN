import sys
import argparse
import gc
import logging
import os

# print("Current Working Directory " , os.getcwd())
# sys.path.append(os.getcwd())

# sys.path.append("/scratch/sz2257/sgan")
sys.path.append("../")
import time
import json
# import yaml
import string
import random

from datetime import datetime
import socket
import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter

from sgan.data.general_loader import data_loader
from sgan.losses import gan_g_loss, gan_d_loss
from training.builders import build_models, build_schedulers, build_optimizers
from training.checkpoint import restore_from_checkpoint, initialize_checkpoint

from training.step import generator_step, discriminator_step
from training.evaluation import check_accuracy
from sgan.utils import int_tuple, bool_flag, get_total_norm

torch.backends.cudnn.benchmark = True
random_str = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
now = datetime.now()
time_str = now.strftime("%Y-%m-%d-%H-%M-%S")
hostname = socket.gethostname()
# time_str="_".join(writer.get_logdir().split("/")[1].split("_")[:2])
# output_dir="/media/felicia/Data/sgan_results/{}".format(time_str)

output_dir="/scratch/sz2257/sgan/sgan_results/{}".format(time_str)

# data_dir='/media/felicia/Data/basketball-partial'
data_dir='/scratch/sz2257/sgan/basketball-partial'

parser = argparse.ArgumentParser()
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

# Dataset options
parser.add_argument('--dataset_name', default='01.02.2016.PHX.at.SAC.new', type=str) #default:zara1
parser.add_argument('--dataset_dir', default=data_dir, type=str)
parser.add_argument('--delim', default=',') #default: ' '
parser.add_argument('--loader_num_workers', default=4, type=int)
parser.add_argument('--obs_len', default=8, type=int)
parser.add_argument('--pred_len', default=8, type=int)
parser.add_argument('--skip', default=1, type=int)
parser.add_argument('--metric', default="meter", type=str)
parser.add_argument("--model", default="baseline", type=str)
parser.add_argument("--schema", default="nfl", type=str)
# Optimization
parser.add_argument('--batch_size', default=128, type=int) #32
parser.add_argument('--num_iterations', default=20000, type=int) #default:10000
parser.add_argument('--num_epochs', default=500, type=int)

# Model Options
parser.add_argument('--embedding_dim', default=16, type=int) #64
parser.add_argument('--num_layers', default=1, type=int)
parser.add_argument('--dropout', default=0, type=float)
parser.add_argument('--batch_norm', default=0, type=bool_flag) #default:0-bool_flag
parser.add_argument('--mlp_dim', default=64, type=int) #default: 1024
parser.add_argument('--interaction_activation', default="none", type=str)

parser.add_argument('--tp_dropout', default=0, type=float)
parser.add_argument('--team_embedding_dim', default=16, type=int) #default: 1024
parser.add_argument('--pos_embedding_dim', default=32, type=int) #default: 1024

# Generator Options
parser.add_argument('--encoder_h_dim_g', default=32, type=int) #default:64
parser.add_argument('--decoder_h_dim_g', default=32, type=int) #default:128
parser.add_argument('--noise_dim', default=(8,), type=int_tuple) # default: None-int_tuple
parser.add_argument('--noise_type', default='gaussian')
parser.add_argument('--noise_mix_type', default='global') #default:pred
parser.add_argument('--clipping_threshold_g', default=1.5, type=float) #default:0
parser.add_argument('--g_learning_rate', default=1e-3, type=float) #default:5e-4,0.001
parser.add_argument('--g_steps', default=1, type=int)
parser.add_argument('--g_gamma', default=0.8, type=float) #default:5e-4, 0.001
# Discriminator Options
parser.add_argument('--d_type', default='local', type=str) #default:'local'
parser.add_argument('--encoder_h_dim_d', default=64, type=int) #default:64
parser.add_argument('--d_learning_rate', default=1e-3, type=float) #default:5e-4, 0.001
parser.add_argument('--d_steps', default=2, type=int) #default:2
parser.add_argument('--clipping_threshold_d', default=0, type=float)
parser.add_argument('--d_activation', default='relu', type=str) # 'relu'
parser.add_argument('--d_gamma', default=0.8, type=float) #default:5e-4, 0.001

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
parser.add_argument('--output_dir', default="../experiments") # os.getcwd()
parser.add_argument('--print_every', default=10, type=int) #default:5
parser.add_argument('--checkpoint_every', default=50, type=int) #default:100
parser.add_argument('--checkpoint_name', default='basketball_phx_sac')
parser.add_argument('--checkpoint_start_from', default=None)
parser.add_argument('--restore_from_checkpoint', default=0, type=int) #default:1
parser.add_argument('--num_samples_check', default=5000, type=int)
parser.add_argument("--tb_path", default=None, type=str)

# Misc
parser.add_argument('--use_gpu', default=1, type=int) # 1: use_gpu
parser.add_argument('--timing', default=0, type=int)
parser.add_argument('--gpu_num', default="0", type=str)


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)


def get_dtypes(args):
    long_dtype = torch.LongTensor
    float_dtype = torch.FloatTensor
    if args.use_gpu == 1:
        long_dtype = torch.cuda.LongTensor
        float_dtype = torch.cuda.FloatTensor
    return long_dtype, float_dtype


def load_schema(schema_path):
    with open(schema_path) as fp:
        return json.load(fp)

def main(args):
    print(args)
    args.checkpoint_name = args.checkpoint_name + "_" + random_str
    tensorboard_name = "_".join([args.checkpoint_name, args.dataset_name, time_str, hostname])
    dataset_runs_dir = os.path.join(args.output_dir, "runs", args.dataset_name)
    dataset_ckpt_dir = os.path.join(args.output_dir, "checkpoints", args.dataset_name)
    if not os.path.exists(dataset_runs_dir):
        os.mkdir(dataset_runs_dir)
    if not os.path.exists(dataset_ckpt_dir):
        os.mkdir(dataset_ckpt_dir)

    tensorboard_path = os.path.join(dataset_runs_dir, tensorboard_name)
    writer = SummaryWriter(tensorboard_path)
    log_path = "{}/config.txt".format(tensorboard_path)
    with open(log_path, "a") as f:
        json.dump(args.__dict__, f, indent=2)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    schema_path = "../sgan/data/configs/{}.json".format(args.schema)
    schema = load_schema(schema_path)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    # train_path = get_dset_path(args.dataset_name, 'train')
    # val_path = get_dset_path(args.dataset_name, 'val')
    train_path= os.path.join(args.dataset_dir,args.dataset_name,'train_sample') # 10 files:0-9
    val_path= os.path.join(args.dataset_dir,args.dataset_name,'val_sample') # 5 files: 10-14

    long_dtype, float_dtype = get_dtypes(args)

    logger.info("Initializing train dataset")
    train_dset, train_loader = data_loader(args, train_path, schema)
    logger.info("Initializing val dataset")
    _, val_loader = data_loader(args, val_path, schema)

    iterations_per_epoch = len(train_dset) / args.batch_size / args.d_steps
    if args.num_epochs:
        args.num_iterations = int(iterations_per_epoch * args.num_epochs)

    logger.info(
        'There are {} iterations per epoch'.format(iterations_per_epoch)
    )

    generator, discriminator = build_models(args, schema, args.model)

    generator.type(float_dtype).train()
    logger.info('Here is the generator:')
    logger.info(generator)

    discriminator.type(float_dtype).train()
    logger.info('Here is the discriminator:')
    logger.info(discriminator)

    g_loss_fn = gan_g_loss
    d_loss_fn = gan_d_loss

    optimizer_g, optimizer_d = build_optimizers(args, generator, discriminator)
    scheduler_g, scheduler_d = build_schedulers(args, optimizer_g, optimizer_d)

    # Maybe restore from checkpoint
    restore_path = None
    if args.checkpoint_start_from is not None:
        restore_path = args.checkpoint_start_from
    elif args.restore_from_checkpoint == 1:
        restore_path = os.path.join(dataset_ckpt_dir,
                                    '%s_with_model.pt' % args.checkpoint_name)

    if restore_path is not None and os.path.isfile(restore_path):
        logger.info('Restoring from checkpoint {}'.format(restore_path))
        checkpoint = torch.load(restore_path)
        generator, discriminator, optimizer_g, optimizer_d, t, epoch = \
            restore_from_checkpoint(checkpoint, generator, discriminator, optimizer_g, optimizer_d)
        checkpoint['restore_ts'].append(t)
    else:
        # Starting from scratch, so initialize checkpoint data structure
        t, epoch = 0, 0
        checkpoint = initialize_checkpoint(args)
    t0 = None
    while t < args.num_iterations:
        gc.collect()
        d_steps_left = args.d_steps
        g_steps_left = args.g_steps
        epoch += 1
        logger.info('Starting epoch {}'.format(epoch))
        scheduler_g.step()
        scheduler_d.step()
        for batch in train_loader:
            if args.timing == 1:
                torch.cuda.synchronize()
                t1 = time.time()

            # Decide whether to use the batch for stepping on discriminator or
            # generator; an iteration consists of args.d_steps steps on the
            # discriminator followed by args.g_steps steps on the generator.
            if d_steps_left > 0:
                step_type = 'd'
                losses_d = discriminator_step(args, batch, generator,
                                              discriminator, d_loss_fn,
                                              optimizer_d)
                checkpoint['norm_d'].append(
                    get_total_norm(discriminator.parameters()))
                d_steps_left -= 1
            elif g_steps_left > 0:
                step_type = 'g'
                losses_g = generator_step(args, batch, generator,
                                          discriminator, g_loss_fn,
                                          optimizer_g)
                checkpoint['norm_g'].append(
                    get_total_norm(generator.parameters())
                )
                g_steps_left -= 1

            if args.timing == 1:
                torch.cuda.synchronize()
                t2 = time.time()
                logger.info('{} step took {}'.format(step_type, t2 - t1))

            # Skip the rest if we are not at the end of an iteration
            if d_steps_left > 0 or g_steps_left > 0:
                continue

            if args.timing == 1:
                if t0 is not None:
                    logger.info('Interation {} took {}'.format(
                        t - 1, time.time() - t0
                    ))
                t0 = time.time()

            # Maybe save loss
            if t % args.print_every == 0:
                logger.info('t = {} / {}'.format(t + 1, args.num_iterations))
                for k, v in sorted(losses_d.items()):
                    # logger.info('  [D] {}: {:.3f}'.format(k, v))
                    checkpoint['D_losses'][k].append(v)
                for k, v in sorted(losses_g.items()):
                    # logger.info('  [G] {}: {:.3f}'.format(k, v))
                    checkpoint['G_losses'][k].append(v)
                checkpoint['losses_ts'].append(t)

                ## log scalars
                for k, v in sorted(losses_d.items()):
                    writer.add_scalar("loss/{}".format(k), v, t)
                for k, v in sorted(losses_g.items()):
                    writer.add_scalar("loss/{}".format(k), v, t)

            # Maybe save a checkpoint
            if t > 0 and t % args.checkpoint_every == 0:
                checkpoint['counters']['t'] = t
                checkpoint['counters']['epoch'] = epoch
                checkpoint['sample_ts'].append(t)

                # Check stats on the validation set
                logger.info('Checking stats on val ...')
                metrics_val = check_accuracy(
                    args, val_loader, generator, discriminator, d_loss_fn
                )
                logger.info('Checking stats on train ...')
                metrics_train = check_accuracy(
                    args, train_loader, generator, discriminator,
                    d_loss_fn, limit=True
                )

                for k, v in sorted(metrics_val.items()):
                    # logger.info('  [val] {}: {:.3f}'.format(k, v))
                    checkpoint['metrics_val'][k].append(v)
                for k, v in sorted(metrics_train.items()):
                    # logger.info('  [train] {}: {:.3f}'.format(k, v))
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
                    checkpoint['g_best_state'] = generator.state_dict()
                    checkpoint['d_best_state'] = discriminator.state_dict()

                if metrics_val['ade_nl'] == min_ade_nl:
                    logger.info('New low for avg_disp_error_nl')
                    checkpoint['best_t_nl'] = t
                    checkpoint['g_best_nl_state'] = generator.state_dict()
                    checkpoint['d_best_nl_state'] = discriminator.state_dict()

                # Save another checkpoint with model weights and
                # optimizer state
                checkpoint['g_state'] = generator.state_dict()
                checkpoint['g_optim_state'] = optimizer_g.state_dict()
                checkpoint['d_state'] = discriminator.state_dict()
                checkpoint['d_optim_state'] = optimizer_d.state_dict()
                # checkpoint_path = os.path.join(
                #     args.output_dir, '{}_with_model_{:06d}.pt'.format(args.checkpoint_name,t)
                # )
                checkpoint_path = os.path.join(dataset_ckpt_dir, '{}_with_model.pt'.format(args.checkpoint_name))
                backup_checkpoint_path = os.path.join(dataset_ckpt_dir, '{}_with_model_backup.pt'.format(args.checkpoint_name))
                logger.info('Saving checkpoint to {}'.format(checkpoint_path))
                torch.save(checkpoint, checkpoint_path)
                torch.save(checkpoint, backup_checkpoint_path)
                logger.info('Done.')

                # Save a checkpoint with no model weights by making a shallow
                # copy of the checkpoint excluding some items

                # checkpoint_path = os.path.join(
                #     args.output_dir, '{}_no_model_{:06d}.pt' .format(args.checkpoint_name,t))

                checkpoint_path = os.path.join(dataset_ckpt_dir,'{}_no_model.pt' .format(args.checkpoint_name))
                logger.info('Saving checkpoint to {}'.format(checkpoint_path))
                key_blacklist = [
                    'g_state', 'd_state', 'g_best_state', 'g_best_nl_state',
                    'g_optim_state', 'd_optim_state', 'd_best_state',
                    'd_best_nl_state'
                ]
                small_checkpoint = {}
                for k, v in checkpoint.items():
                    if k not in key_blacklist:
                        small_checkpoint[k] = v
                torch.save(small_checkpoint, checkpoint_path)
                logger.info('Done.')

            t += 1
            d_steps_left = args.d_steps
            g_steps_left = args.g_steps
            if t >= args.num_iterations:
                break


        ## log scalars
        # for k, v in sorted(losses_d.items()):
        #     writer.add_scalar("train/{}".format(k),v,epoch)
        # for k, v in sorted(losses_g.items()):
        #     writer.add_scalar("train/{}".format(k),v,epoch)
    writer.flush()






if __name__ == '__main__':
    args = parser.parse_args()

    # log_path="{}/config.yaml".format(writer.get_logdir())
    # with open(log_path,'w') as file:
    #     args_file=yaml.dump(args,file)
    # print(args_file)

    main(args)


