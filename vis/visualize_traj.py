import os
import numpy as np
from tqdm import tqdm

import plotly.graph_objects as go
import matplotlib.pyplot as plt



model_dir='/media/felicia/Data/sgan_results/best_models'
models=[
    'sm.baseline_attention_v3.6.d5.e16.dg10.gg10_with_model.pt',
    'sm.team_pos_attentiontp_v3.6.d5.e16.pe16.te4.tpd5.gg10.dg10.l10_with_model.pt',
    'cs05.baseline_attention_v3.6.d5.e16.dg10.gg10_with_model.pt',
    'cs05.team_pos_attentiontp_v3.6.d0.e16.pe16.te4.tpd5.gg8.dg8.l10_with_model.pt'
]
model_name=models[0]
print('Model: {}'.format(model_name))

sample_dir=os.path.join('/media/felicia/Data/sgan_results/best_samples/n1','{}_test.np'.format(model_name))
pred_dict = np.load(sample_dir, allow_pickle=True).item()

obs_traj_np=pred_dict['obs_traj']
pred_traj_gt_np=pred_dict['pred_traj_gt']
pred_traj_fake_np=pred_dict['pred_traj_fake']
seq_start_end_np=pred_dict['start_end']
pos_vec_np=pred_dict['pos']
team_vec_np=pred_dict['team']
ade_np=pred_dict['ade']
fde_np=pred_dict['fde']


ntraj=obs_traj_np.shape[1]


def vector_color(pos_vec,team_vec):
    """
    Args:
        pos_vec: C F G ball
        team_vec:  0 1 ball
    Returns:
    """
    team_id=np.where(team_vec==1)[1].tolist()

    color_obs=[t*2 for t in team_id]
    color_pred=[t*2+1 for t in team_id]
    return None, team_id,color_obs,color_pred


vis_dir='/media/felicia/Data/sgan_results/vis/'

# import matplotlib.colors as mcolors
#
# def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=-1):
#     if n == -1:
#         n = cmap.N
#     new_cmap = mcolors.LinearSegmentedColormap.from_list(
#          'trunc({name},{a:.2f},{b:.2f})'.format(name=cmap.name, a=minval, b=maxval),
#          cmap(np.linspace(minval, maxval, n)))
#     return new_cmap

cm = plt.cm.get_cmap('tab20')
# newCm=truncate_colormap(plt.get_cmap("inferno"), 0, 6)
fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(111)

for i in tqdm(range(ntraj)):
    for j in range(11):
        obs_x,obs_y=obs_traj_np[:,i,j,0],obs_traj_np[:,i,j,1] # 8
        pred_x,pred_y=pred_traj_fake_np[:,i,j,0],pred_traj_fake_np[:,i,j,1] # 8
        gt_x,gt_y=pred_traj_fake_np[:,i,j,0],pred_traj_gt_np[:,i,j,1] # 8

        _,team_id,color_obs,color_pred=vector_color(None,team_vec_np[:,i,j])
        # print(team_id[0],color_obs[0],color_pred[0])
        plt.scatter(obs_x, obs_y, c=color_obs,vmin=0, vmax=20, cmap=cm, alpha=1,marker='o')
        # plt.scatter(gt_x, gt_y, c=color_obs, vmin=0, vmax=20,cmap=cm, alpha=0.3,marker='o')
        plt.scatter(pred_x, pred_y, c=color_pred, vmin=0, vmax=20,cmap=cm, alpha=1,marker='v')

    plt.colorbar()
    # plt.show()
    plt.savefig(vis_dir + 'model{:01d}_traj{:05d}_1.png'.format(0, i))
    break


# plt.show()


import plotly.express as px
# fig = px.scatter()
fig=go.Figure()

colors_set3=px.colors.qualitative.Set3
colors=[]

for i in tqdm(range(ntraj)):
    for j in range(11):
        obs_x,obs_y=obs_traj_np[:,i,j,0],obs_traj_np[:,i,j,1] # 8
        pred_x,pred_y=pred_traj_fake_np[:,i,j,0],pred_traj_fake_np[:,i,j,1] # 8
        gt_x,gt_y=pred_traj_fake_np[:,i,j,0],pred_traj_gt_np[:,i,j,1] # 8

        _,team_id,color_obs,color_pred=vector_color(None,team_vec_np[:,i,j])
        color_obs_=[colors[c] for c in color_obs]
        color_pred_=[colors[c] for c in color_pred]

        # x=obs_x.tolist()+pred_x.tolist()
        # y=obs_y.tolist()+pred_y.tolist()
        # c=color_obs+color_pred
        # fig=px.scatter(x, y,color=c)
        fig.add_trace(go.Scatter(x=obs_x, y=obs_y, mode='markers',marker=dict(color=color_obs_,size=10)))
        fig.add_trace(go.Scatter(x=pred_x, y=pred_y, mode='markers', marker=dict(color=color_pred_,size=10)))
    break

fig.show()
