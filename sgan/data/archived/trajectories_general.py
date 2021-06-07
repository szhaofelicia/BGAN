import logging
import os
import math
from tqdm import tqdm

import numpy as np

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def seq_collate(data):
    (obs_seq_list, pred_seq_list, obs_seq_rel_list, pred_seq_rel_list,
     obs_team_vec_list, obs_pos_vec_list, pred_team_vec_list, pred_pos_vec_list,
     non_linear_ped_list, loss_mask_list) = zip(*data)

    _len = [len(seq) for seq in obs_seq_list]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end]
                     for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    # Data format: batch, input_size, seq_len
    # LSTM input format: seq_len, batch, input_size
    obs_traj = torch.cat(obs_seq_list, dim=0).permute(2, 0, 1)
    pred_traj = torch.cat(pred_seq_list, dim=0).permute(2, 0, 1)
    obs_traj_rel = torch.cat(obs_seq_rel_list, dim=0).permute(2, 0, 1)
    pred_traj_rel = torch.cat(pred_seq_rel_list, dim=0).permute(2, 0, 1)

    obs_team_vec = torch.cat(obs_team_vec_list, dim=0).permute(2, 0, 1)
    obs_pos_vec = torch.cat(obs_pos_vec_list, dim=0).permute(2, 0, 1)
    pred_team_vec = torch.cat(pred_team_vec_list, dim=0).permute(2, 0, 1)
    pred_pos_vec = torch.cat(pred_pos_vec_list, dim=0).permute(2, 0, 1)

    non_linear_ped = torch.cat(non_linear_ped_list)
    loss_mask = torch.cat(loss_mask_list, dim=0)
    seq_start_end = torch.LongTensor(seq_start_end)
    out = [
        obs_traj, pred_traj, obs_traj_rel, pred_traj_rel,
        obs_team_vec, obs_pos_vec, pred_team_vec, pred_pos_vec,
        non_linear_ped, loss_mask, seq_start_end
    ]

    return tuple(out)


def read_file(_path, delim='\t'):
    lines = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        next(f)
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) if isfloat(i) else i for i in line]
            lines.append(line)
    return lines


def parse_file(_path, delim='\t',dset="basketball",trajD=2):
    """
    basketball: index   frame_id    team_id player_id   pos_x   pos_y   player_position
    csgo:   index   frame_id    team_id player_id   pos_x   pos_y   pos_z
    dota:   index   frame_id    team_id player_id   pos_x   pos_y
    nfl:    index    frame_id    team_id player_id   pos_x   pos_y   player_position
    """

    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    lines = read_file(_path, delim)
    team_ids_ = np.unique([int(line[2]) for line in lines if isfloat(line[2])]).tolist()
    team_ids={
        "basketball": team_ids_+["ball"],
        "csgo": ["T","CT"],
        "dota": [0,1],
        "nfl": [0,1,"ball"],
    }
    team_dim=len(team_ids[dset])
    posi_ids = {
        "basketball":["C", "F", "G", "ball"],
        "csgo":[None], # no player_position
        "dota":[None], # no player_position
        "nfl":["player","ball"]
    }
    posi_dim=len(posi_ids[dset])

    index,frame_idx,team_idx, player_idx,traj_idx=0,1,2,3,4
    posi_idx=4+trajD if dset=="basketball" or dset=="nfl" else None

    for line in lines:
        row = []
        team_vector = [0.0] * team_dim
        pos_vector = [0.0] * posi_dim
        for col, value in enumerate(line):
            if col==index or col== frame_idx or traj_idx<=col<traj_idx+trajD: # index
                row.append(value)
            elif col == team_idx:  # team_id
                team = team_ids[dset].index(int(value) if isfloat(value) else value)
                team_vector[team] = 1.0
            elif col == player_idx:  # player_id
                if value == "ball":
                    row.append(-1.0)
                else:
                    row.append(value)  # float
            elif col == posi_idx :  # player_position: basketball,nfl
                positions = value.strip('"').split(",")
                for pos in positions:
                    pos_vector[posi_ids[dset].index(pos)] = 1.0
        row += team_vector  # team_id
        row += pos_vector  # player_position

        data.append(row)
    return np.asarray(data)


def poly_fit(traj, traj_len, threshold,trajD=2):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    res_z = np.polyfit(t, traj[2, -traj_len:], 2, full=True)[1] if trajD>2 else 0.0

    if res_x + res_y + res_z >= threshold:
        return 1.0
    else:
        return 0.0


class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""

    def __init__(
            self, data_dir, obs_len=8, pred_len=12, skip=1, threshold=0.002,
            min_ped=1, delim='\t', metric="meter", dset="basketball",trajD=2
    ):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        - metric: trajectory metric, meter or feet

        columns in csv file:
        (idx), frame_id,team_id,player_id,pos_x, pos_y, player_position
        ->
        data:
        idx, frame_id,player_id,pos_x, pos_y, team_vector,position_vector

        """
        super(TrajectoryDataset, self).__init__()

        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim
        self.dset=dset
        self.trajD=trajD

        team_ids={
            "basketball":[0,1,"ball"],
            "csgo": ["T","CT"],
            "dota": [0,1],
            "nfl": [0,1,"ball"],
        }
        posi_ids = {
            "basketball":["C", "F", "G", "ball"],
            "csgo":[None], # no player_position
            "dota":[None], # no player_position
            "nfl":["player","ball"]
        }
        self.team_dim=len(team_ids[self.dset])
        self.posi_dim=len(posi_ids[self.dset])

        # if metric=="meter":
        #     self.factor=0.3048 # foot to meter
        # else:
        #     self.factor=1.0 # foot to foot

        # convert other metrics to meter
        if metric=="meter" or metric=="original":
            self.factor=1.0 # original metric
        elif metric=="foot":
            self.factor=0.3048 # foot to meter
        elif metric=="yard":
            self.factor=0.9144 # yard to meter

        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        loss_mask_list = []
        non_linear_ped = []

        team_vec_list = []
        pos_vec_list = []

        frame_idx,player_idx,traj_idx,team_idx,posi_idx=1,2,3,3+self.trajD,3+self.trajD+self.team_dim


        for path in tqdm(all_files):
            data = parse_file(path, delim, dset=self.dset,trajD=self.trajD)

            frames = np.unique(data[:, 0]).tolist()
            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, frame_idx], :])  # frame_id
            num_sequences = int(
                math.ceil((len(frames) - self.seq_len + 1) / skip))

            for idx in range(0, num_sequences * self.skip + 1, skip):
                curr_seq_data = np.concatenate(
                    frame_data[idx:idx + self.seq_len], axis=0)
                peds_in_curr_seq = np.unique(curr_seq_data[:, player_idx])  # player_id
                curr_seq_rel = np.zeros((len(peds_in_curr_seq), self.trajD,self.seq_len))
                curr_seq = np.zeros((len(peds_in_curr_seq), self.trajD, self.seq_len))
                curr_loss_mask = np.zeros((len(peds_in_curr_seq),self.seq_len))
                # vectors
                curr_team = np.zeros((len(peds_in_curr_seq), self.team_dim, self.seq_len))
                curr_position = np.zeros((len(peds_in_curr_seq), self.posi_dim, self.seq_len))

                num_peds_considered = 0
                _non_linear_ped = []


                for _, ped_id in enumerate(peds_in_curr_seq):
                    curr_ped_seq_full = curr_seq_data[curr_seq_data[:, player_idx] == ped_id, :]  # player_id
                    curr_ped_seq_full = np.around(curr_ped_seq_full, decimals=4)
                    pad_front = frames.index(curr_ped_seq_full[0, frame_idx]) - idx  # frame_id
                    pad_end = frames.index(curr_ped_seq_full[-1, frame_idx]) - idx + 1  # frame_id
                    if pad_end - pad_front != self.seq_len or curr_ped_seq_full.shape[0] != self.seq_len:
                        continue
                    curr_ped_seq = np.transpose(curr_ped_seq_full[:, traj_idx:team_idx])  # x,y,(z)
                    curr_ped_seq = curr_ped_seq * self.factor # conversion
                    # Make coordinates relative
                    rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                    rel_curr_ped_seq[:, 1:] = curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
                    _idx = num_peds_considered

                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
                    # Linear vs Non-Linear Trajectory
                    _non_linear_ped.append(
                        poly_fit(curr_ped_seq, pred_len, threshold,trajD=self.trajD))
                    curr_loss_mask[_idx, pad_front:pad_end] = 1

                    # Team vector
                    curr_ped_team = np.transpose(curr_ped_seq_full[:, team_idx:posi_idx])  # team vector
                    curr_team[_idx, :, pad_front:pad_end] = curr_ped_team

                    # Position Vector
                    curr_ped_pos = np.transpose(curr_ped_seq_full[:, posi_idx:])  # position vector
                    curr_position[_idx, :, pad_front:pad_end] = curr_ped_pos

                    num_peds_considered += 1

                if num_peds_considered > min_ped:
                    non_linear_ped += _non_linear_ped
                    num_peds_in_seq.append(num_peds_considered)
                    loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                    seq_list.append(curr_seq[:num_peds_considered])
                    seq_list_rel.append(curr_seq_rel[:num_peds_considered])
                    team_vec_list.append(curr_team[:num_peds_considered])  # team vector
                    pos_vec_list.append(curr_position[:num_peds_considered])  # pos_vec_list

        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)

        team_vec_list = np.concatenate(team_vec_list, axis=0)
        pos_vec_list = np.concatenate(pos_vec_list, axis=0)

        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_ped = np.asarray(non_linear_ped)

        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.from_numpy(
            seq_list[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj = torch.from_numpy(
            seq_list[:, :, self.obs_len:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, self.obs_len:]).type(torch.float)

        self.obs_team_vec = torch.from_numpy(
            team_vec_list[:, :, :self.obs_len]).type(torch.float)
        self.obs_pos_vec = torch.from_numpy(
            pos_vec_list[:, :, :self.obs_len]).type(torch.float)

        self.obs_team_vec_pred = torch.from_numpy(
            team_vec_list[:, :, self.obs_len:]).type(torch.float)
        self.obs_pos_vec_pred = torch.from_numpy(
            pos_vec_list[:, :, self.obs_len:]).type(torch.float)

        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]


    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        out = [
            self.obs_traj[start:end, :], self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
            self.obs_team_vec[start:end, :], self.obs_pos_vec[start:end, :],
            self.obs_team_vec_pred[start: end, :], self.obs_pos_vec_pred[start: end, :],
            self.non_linear_ped[start:end], self.loss_mask[start:end, :]
        ]
        return out
