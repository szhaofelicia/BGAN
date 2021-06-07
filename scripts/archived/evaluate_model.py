import argparse
import os
import torch

from attrdict import AttrDict

from sgan.data.loader import data_loader
from sgan.losses import displacement_error, final_displacement_error
from sgan.utils import relative_to_abs, get_dset_path
from sgan.utils import int_tuple, bool_flag, get_total_norm

# from sgan.models_linear import TrajectoryLinearRegressor

# from sgan.models import TrajectoryGenerator
# from sgan.models_old import TrajectoryGenerator
from sgan.models_teampos import TrajectoryGenerator


parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str,
                    default="/media/felicia/Data/sgan_results/models/Single-Match/Team_Pos_With_TP_Attention/models")  # default:"models/sgan-models"
# default = "/media/felicia/Data/sgan_results/May03_03-05-44/model")  # default:"models/sgan-models"
parser.add_argument('--num_samples', default=1, type=int)  # N=20
parser.add_argument('--dset_type', default='test_sample', type=str)
parser.add_argument('--dataset_dir', default='/media/felicia/Data/basketball-partial', type=str)

# # Dataset options
# parser.add_argument('--delim', default='\t') #default: ' '
# parser.add_argument('--loader_num_workers', default=4, type=int)
# parser.add_argument('--obs_len', default=8, type=int)
# parser.add_argument('--pred_len', default=8, type=int)
# parser.add_argument('--skip', default=1, type=int)
# parser.add_argument('--metric', default="meter", type=str)
# # parser.add_argument("--model", default="baseline", type=str)
#
# # Optimization
# parser.add_argument('--batch_size', default=32, type=int) #32
# parser.add_argument('--num_iterations', default=20000, type=int) #default:10000
# parser.add_argument('--num_epochs', default=500, type=int)
#
# # Model Options
# parser.add_argument('--embedding_dim', default=16, type=int) #64
# parser.add_argument('--num_layers', default=1, type=int)
# parser.add_argument('--dropout', default=0, type=float)
# parser.add_argument('--tp_dropout', default=0.5, type=float)
# parser.add_argument('--batch_norm', default=0, type=bool_flag) #default:0-bool_flag
# parser.add_argument('--mlp_dim', default=128, type=int) #default: 1024
# parser.add_argument('--team_embedding_dim', default=4, type=int) #default: 1024
# parser.add_argument('--pos_embedding_dim', default=16, type=int) #default: 1024
# parser.add_argument('--interaction_activation', default="none", type=str)
#
# # Generator Options
# parser.add_argument('--encoder_h_dim_g', default=32, type=int) #default:64
# parser.add_argument('--decoder_h_dim_g', default=32, type=int) #default:128
# parser.add_argument('--noise_dim', default=(8,), type=int_tuple) # default: None-int_tuple
# parser.add_argument('--noise_type', default='gaussian')
# parser.add_argument('--noise_mix_type', default='global') #default:pred
# parser.add_argument('--clipping_threshold_g', default=1.5, type=float) #default:0
# parser.add_argument('--g_learning_rate', default=1e-3, type=float) #default:5e-4,0.001
# parser.add_argument('--g_steps', default=1, type=int)
#
# # Discriminator Options
# parser.add_argument('--d_type', default='local', type=str) #default:'local'
# parser.add_argument('--encoder_h_dim_d', default=64, type=int) #default:64
# parser.add_argument('--d_learning_rate', default=1e-3, type=float) #default:5e-4, 0.001
# parser.add_argument('--d_steps', default=2, type=int) #default:2
# parser.add_argument('--clipping_threshold_d', default=0, type=float)
# parser.add_argument('--d_activation', default='relu', type=str) # 'relu'
#
#
# # Pooling Options
# parser.add_argument('--pooling_type', default='none') #default:'pool_net', none for lstm
# parser.add_argument('--pool_every_timestep', default=0, type=bool_flag) #default:1-bool_flag
#
# # Pool Net Option
# parser.add_argument('--bottleneck_dim', default=32, type=int) # 1024
#
# # Social Pooling Options
# parser.add_argument('--neighborhood_size', default=2.0, type=float)
# parser.add_argument('--grid_size', default=8, type=int)
#
# # Loss Options
# parser.add_argument('--l2_loss_weight', default=1, type=float) #default:0->1
# parser.add_argument('--best_k', default=10, type=int) #default:1
# parser.add_argument('--l2_loss_mode', default="raw", type=str) #default:"raw"



def get_generator(checkpoint):
    args = AttrDict(checkpoint['args'])

    args.dataset_dir='/media/felicia/Data/basketball-partial'

    generator = TrajectoryGenerator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=args.embedding_dim,
        encoder_h_dim=args.encoder_h_dim_g,
        decoder_h_dim=args.decoder_h_dim_g,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        noise_dim=args.noise_dim,
        noise_type=args.noise_type,
        noise_mix_type=args.noise_mix_type,
        pooling_type=args.pooling_type,
        pool_every_timestep=args.pool_every_timestep,
        dropout=args.dropout,
        tp_dropout=args.tp_dropout,
        bottleneck_dim=args.bottleneck_dim,
        neighborhood_size=args.neighborhood_size,
        grid_size=args.grid_size,
        batch_norm=args.batch_norm,
        team_embedding_dim=args.team_embedding_dim,
        pos_embedding_dim=args.pos_embedding_dim,
        interaction_activation=args.interaction_activation
    )
    # generator.load_state_dict(checkpoint['g_state'])
    generator.load_state_dict(checkpoint['g_best_state'])
    generator.cuda()
    generator.train()
    return generator

    # regressor = TrajectoryLinearRegressor(
    #     obs_len=args.obs_len,
    #     pred_len=args.pred_len,
    #     embedding_dim=args.embedding_dim,
    #     mlp_dim=args.mlp_dim,
    #     dropout=args.dropout,
    #     batch_norm=args.batch_norm,
    # )
    # regressor.load_state_dict(checkpoint['r_state'])
    # regressor.cuda()
    # regressor.train()
    # return regressor


def evaluate_helper(error, seq_start_end):
    sum_ = 0
    error = torch.stack(error, dim=1)

    for (start, end) in seq_start_end:
        start = start.item()
        end = end.item()
        _error = error[start:end]
        _error = torch.sum(_error, dim=0)
        _error = torch.min(_error)
        sum_ += _error
    return sum_


def evaluate(args, loader, generator, num_samples):
    ade_outer, fde_outer = [], []
    total_traj = 0
    with torch.no_grad():
        ade_error, fde_error = [], []

        for batch in loader:
            batch = [tensor.cuda() for tensor in batch]
            # print(len(batch))
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
             obs_team_vec, obs_pos_vec, pred_team_vec, pred_pos_vec,
             non_linear_ped, loss_mask, seq_start_end) = batch

            ade, fde = [], []
            total_traj += pred_traj_gt.size(1)

            for _ in range(num_samples):
                # pred_traj_fake_rel = generator(obs_traj, obs_traj_rel, seq_start_end) #regressor
                pred_traj_fake_rel = generator(obs_traj, obs_traj_rel, seq_start_end, obs_team_vec, obs_pos_vec) # generator

                pred_traj_fake = relative_to_abs(
                    pred_traj_fake_rel, obs_traj[-1]
                )
                ade.append(displacement_error(
                    pred_traj_fake, pred_traj_gt, mode='raw'
                ))
                fde.append(final_displacement_error(
                    pred_traj_fake[-1], pred_traj_gt[-1], mode='raw'
                ))

            # ade_error.append(ade.item())
            # fde_error.append(fde.item())

            ade_sum = evaluate_helper(ade, seq_start_end)
            fde_sum = evaluate_helper(fde, seq_start_end)

            ade_outer.append(ade_sum)
            fde_outer.append(fde_sum)

        ade = sum(ade_outer) / (total_traj * args.pred_len)
        fde = sum(fde_outer) / (total_traj)

        # print(len(ade_error),len(ade_error[0]))
        # print(ade_error)

        # print(sum(ade_error),sum(sum(ade_error)))
        # print(total_traj,args.pred_len)
        # ade_=sum(sum(ade_error)) / (total_traj * args.pred_len)
        # fde_=sum(sum(fde_error))/ (total_traj * args.pred_len)
        #
        # print('ADE',ade_)
        # print('FDE',fde_)


        return ade, fde


def main(args):
    if os.path.isdir(args.model_path):
        filenames = os.listdir(args.model_path)
        filenames.sort()
        paths = [
            os.path.join(args.model_path, file_) for file_ in filenames
        ]
    else:
        paths = [args.model_path]

    count=0
    for path in paths:
        model_name=path.split('/')[-1]
        print('Model: {}'.format(model_name))
        checkpoint = torch.load(path)
        generator = get_generator(checkpoint)
        _args = AttrDict(checkpoint['args'])

        print(_args.dataset_name)

        # if _args.dataset_name=="zara1":
        #     for k in _args:
        #         print(k,_args[k])

        # path = get_dset_path(_args.dataset_name, args.dset_type)

        path = os.path.join(args.dataset_dir, _args.dataset_name, 'test_sample')  # 10 files:0-9
        _, loader = data_loader(_args, path)
        ade, fde = evaluate(_args, loader, generator, args.num_samples)
        print('Dataset: {}, Pred Len: {}, ADE: {:.2f}, FDE: {:.2f}'.format(
            _args.dataset_name, _args.pred_len, ade, fde))
        count+=1



if __name__ == '__main__':

    args = parser.parse_args()
    print(args)
    main(args)


"""
source env/bin/activate 
ipython scripts/evaluate_model.py 

"""
