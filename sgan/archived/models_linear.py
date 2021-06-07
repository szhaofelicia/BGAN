import torch
import torch.nn as nn


def make_mlp(dim_list, activation='relu', batch_norm=True, dropout=0):
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim_out))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)


def get_noise(shape, noise_type):
    if noise_type == 'gaussian':
        return torch.randn(*shape).cuda()
    elif noise_type == 'uniform':
        return torch.rand(*shape).sub_(0.5).mul_(2.0).cuda()
    raise ValueError('Unrecognized noise type "%s"' % noise_type)



class TrajectoryLinearRegressor(nn.Module):
    def __init__(
        self,obs_len, pred_len, embedding_dim=64,
        mlp_dim=1024,activation='relu', batch_norm=True, dropout=0.0,
    ):

        super(TrajectoryLinearRegressor, self).__init__()

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.mlp_dim = mlp_dim
        self.embedding_dim = embedding_dim

        dim_list=[
            obs_len*2, mlp_dim, pred_len*2
        ]

        self.regressor=make_mlp(
            dim_list,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout
        )

    def forward(self, obs_traj, obs_traj_rel, seq_start_end, user_noise=None):
        batch = obs_traj_rel.size(1)

        # last_pos = obs_traj[-1]
        # last_pos_rel = obs_traj_rel[-1]

        input=obs_traj_rel.reshape(-1,self.obs_len*2)
        output=self.regressor(input)
        pred_traj_fake_rel=output.reshape(-1,batch,2)

        return pred_traj_fake_rel




