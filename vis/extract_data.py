import os
import torch
from torch.utils.data import DataLoader

from tqdm import tqdm
import numpy as np
from attrdict import AttrDict

from sgan.data.loader import data_loader
from sgan.losses import displacement_error, final_displacement_error
from sgan.utils import relative_to_abs, get_dset_path


# from sgan.models import TrajectoryGenerator
# from sgan.models_old import TrajectoryGenerator
from sgan.models_teampos import TrajectoryGenerator

from sgan.data.trajectories_basketball_0427 import TrajectoryDataset, seq_collate


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
    generator.load_state_dict(checkpoint['g_state'])
    generator.cuda()
    generator.train()
    return generator


model_dir='/media/felicia/Data/sgan_results/best_models'
models=[
    'sm.baseline_attention_v3.6.d5.e16.dg10.gg10_with_model.pt',
    'sm.team_pos_attentiontp_v3.6.d5.e16.pe16.te4.tpd5.gg10.dg10.l10_with_model.pt',
    'cs05.baseline_attention_v3.6.d5.e16.dg10.gg10_with_model.pt',
    'cs05.team_pos_attentiontp_v3.6.d0.e16.pe16.te4.tpd5.gg8.dg8.l10_with_model.pt'
]


"""
0:same-sgan+attention
1: same-sgan+both
2:cross-sgan+attention
3:cross-sgan+both
"""

model_name=models[3]

path=os.path.join(model_dir,model_name)
print('Model: {}'.format(model_name))

checkpoint = torch.load(path)
generator = get_generator(checkpoint)
_args = AttrDict(checkpoint['args'])

print('Dataset: {}'.format(_args.dataset_name))


BATCH=1
dataset_dir='/media/felicia/Data/basketball-partial'
path = os.path.join(dataset_dir, _args.dataset_name, 'test_sample')  # 10 files:0-9

dset = TrajectoryDataset(
        path,
        obs_len=_args.obs_len,
        pred_len=_args.pred_len,
        skip=_args.skip,
        delim=_args.delim,
        metric=_args.metric
    )

loader = DataLoader(
        dset,
        batch_size=BATCH,
        shuffle=False,
        num_workers=_args.loader_num_workers,
        collate_fn=seq_collate
    )


obs_traj_list=[]
pred_traj_gt_list=[]
pred_traj_fake_list=[]
seq_start_end_list=[]
pos_vec_list=[]
team_vec_list=[]
ade_list=[]
fde_list=[]


with torch.no_grad():
    for batch in tqdm(loader):
        batch = [tensor.cuda() for tensor in batch]
        (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
         obs_team_vec, obs_pos_vec, pred_team_vec, pred_pos_vec,
         non_linear_ped, loss_mask, seq_start_end) = batch

        pred_traj_fake_rel = generator(obs_traj, obs_traj_rel, seq_start_end, obs_team_vec, obs_pos_vec)  # generator
        pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

        # print(seq_start_end)
        start,end=seq_start_end[0][0],seq_start_end[0][1]
        if end-start!=11:
            continue

        ade=displacement_error(
            pred_traj_fake, pred_traj_gt, mode='raw'
        )  # batch*11

        fde=final_displacement_error(
            pred_traj_fake[-1], pred_traj_gt[-1], mode='raw'
        ) # batch*11

        obs_traj_list.append(obs_traj.cpu().numpy().reshape(8,-1,11,2)) # obs_len * (batch*11) * 2
        pred_traj_gt_list.append(pred_traj_gt.cpu().numpy().reshape(8,-1,11,2))
        pred_traj_fake_list.append(pred_traj_fake.cpu().numpy().reshape(8,-1,11,2))
        pos_vec_list.append(obs_pos_vec.cpu().numpy().reshape(8,-1,11,4))
        team_vec_list.append(obs_team_vec.cpu().numpy().reshape(8,-1,11,3))

        seq_start_end_list.append(seq_start_end.cpu().numpy()) # Batch * 2

        ade_list.append(ade.cpu().numpy().reshape(-1,11))
        fde_list.append(fde.cpu().numpy().reshape(-1,11))


obs_traj_list=np.concatenate(obs_traj_list,axis=1) # obs_len*batch *11 *
pred_traj_gt_list=np.concatenate(pred_traj_gt_list,axis=1)
pred_traj_fake_list=np.concatenate(pred_traj_fake_list,axis=1)
pos_vec_list=np.concatenate(pos_vec_list,axis=1)
team_vec_list=np.concatenate(team_vec_list,axis=1)

seq_start_end_list=np.concatenate(seq_start_end_list,axis=0)
ade_list=np.concatenate(ade_list,axis=0) # batch *11
fde_list=np.concatenate(fde_list,axis=0) # batch *11

print(obs_traj_list.shape)

# key-sample
testset_dict={
    'obs_traj':obs_traj_list,
    'pred_traj_gt':pred_traj_gt_list,
    'pred_traj_fake':pred_traj_fake_list,
    'start_end':seq_start_end_list,
    'pos':pos_vec_list,
    'team':team_vec_list,
    'ade':ade_list,
    'fde':fde_list
}

save_dir=os.path.join('/media/felicia/Data/sgan_results/best_samples/','{}_test.np'.format(model_name))
with open(save_dir,'wb') as file:
    np.save(file,testset_dict)


"""

sm.baseline_attention_v3.6.d5.e16.dg10.gg10_with_model.pt
sm.team_pos_attentiontp_v3.6.d5.e16.pe16.te4.tpd5.gg10.dg10.l10_with_model.pt

cs05.baseline_attention_v3.6.d5.e16.dg10.gg10_with_model.pt
cs05.team_pos_attentiontp_v3.6.d0.e16.pe16.te4.tpd5.gg8.dg8.l10_with_model.pt
"""