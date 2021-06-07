import os
import numpy as np
from tqdm import tqdm
import plotly.express as px
import plotly.graph_objects as go


def draw_plotly_whole_court(fig, fig_height=600, margins=10):
    # From: https://community.plot.ly/t/arc-shape-with-path/7205/5
    def ellipse_arc(x_center=52.5, y_center=250, a=10.5, b=10.5, start_angle=0.0, end_angle=2 * np.pi, N=200, closed=False):
        t = np.linspace(start_angle, end_angle, N)
        x = x_center + a * np.cos(t)
        y = y_center + b * np.sin(t)
        path = f'M {x[0]}, {y[0]}'
        for k in range(1, len(t)):
            path += f'L{x[k]}, {y[k]}'
        if closed:
            path += ' Z'
        return path

    fig_width = fig_height * (470*2 + 2 * margins) / (500 + 2 * margins)
    fig.update_layout(width=fig_width, height=fig_height)

    fig.update_xaxes(range=[- margins, 470*2 + margins])
    fig.update_yaxes(range=[- margins, 500 + margins])


    threept_break_y = 89.47765084
    three_line_col = "#777777"
    main_line_col = "#777777"

    fig.update_layout(
        # Line Horizontal
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="white",
        plot_bgcolor="white",
        yaxis=dict(
            scaleanchor="x",
            scaleratio=1,
            showgrid=False,
            zeroline=False,
            showline=False,
            ticks='',
            showticklabels=False,
            fixedrange=True,
        ),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showline=False,
            ticks='',
            showticklabels=False,
            fixedrange=True,
        ),

        # width:500, height: 470
        # x:-52.5, y:-250
        shapes=[
            dict(
                type="rect", y0=0, x0=0, y1=500, x1=470*2,
                line=dict(color=main_line_col, width=1),
                # fillcolor='#333333',
                layer='below'
            ),  ## sideline rect
            dict(
                type="line", y0=0, x0=470, y1=500, x1=470,
                line=dict(color=main_line_col, width=1),
                layer='below'
            ), # half-court line

            dict(
                type="circle", y0=190, x0=410, y1=310, x1=530, xref="x", yref="y",
                line=dict(color=main_line_col, width=1),
                # fillcolor='#dddddd',
                layer='below'
            ),  # center circle: radius=60,x=77.5->122.5

            dict(
                type="rect", y0=170, x0=0, y1=330, x1=190,
                line=dict(color=main_line_col, width=1),
                # fillcolor='#333333',
                layer='below'
            ),# lane line rect
            dict(
                type="rect", y0=190, x0=0, y1=310, x1=190,
                line=dict(color=main_line_col, width=1),
                # fillcolor='#333333',
                layer='below'
            ), # foul line rect
            dict(
                type="circle", y0=190, x0=130, y1=310, x1=250, xref="x", yref="y",
                line=dict(color=main_line_col, width=1),
                # fillcolor='#dddddd',
                layer='below'
            ), # free-throw circle
            dict(
                type="line", y0=190, x0=190, y1=310, x1=190,
                line=dict(color=main_line_col, width=1),
                layer='below'
            ), # foul line

            dict(
                type="rect", y0=248, x0=40, y1=252, x1=45.25,
                line=dict(color="#ec7607", width=1),
                fillcolor='#ec7607',
            ), # hoop rect,r=5.25
            dict(
                type="circle", y0=242.5, x0=45, y1=257.5, x1=60, xref="x", yref="y",
                line=dict(color="#ec7607", width=1),
            ), # hoop circle
            dict(
                type="line", y0=220, x0=40, y1=280, x1=40,
                line=dict(color="#ec7607", width=1),
            ), # backboard

            dict(type="path",
                 path=ellipse_arc(a=40, b=40, start_angle=-np.pi*0.5, end_angle=np.pi*0.5),
                 line=dict(color=main_line_col, width=1), layer='below'), # no-change semi-circle
            dict(type="path",
                 path=ellipse_arc(a=237.5, b=237.5, start_angle=0.386283101-np.pi*0.5, end_angle=np.pi*0.5 - 0.386283101),
                 line=dict(color=main_line_col, width=1), layer='below'), # three-point line:arc
            dict(
                type="line", y0=30, x0=0, y1=30, x1=threept_break_y+52.5,
                line=dict(color=three_line_col, width=1), layer='below'
            ), # three-point line:left edge
            # dict(
            #     type="line", x0=-220, y0=-52.5, x1=-220, y1=threept_break_y,
            #     line=dict(color=three_line_col, width=1), layer='below'
            # ),
            dict(
                type="line", y0=470, x0=0, y1=470, x1=threept_break_y+52.5,
                line=dict(color=three_line_col, width=1), layer='below'
            ), # three-point line:right edge

            dict(
                type="line", y0=0, x0=280, y1=30, x1=280,
                line=dict(color=main_line_col, width=1), layer='below'
            ), # midcourt area marker:left
            dict(
                type="line", y0=520, x0=280, y1=470, x1=280,
                line=dict(color=main_line_col, width=1), layer='below'
            ), # midcourt area marker:right
            dict(
                type="line", y0=160, x0=70, y1=170, x1=70,
                line=dict(color=main_line_col, width=1), layer='below'
            ), # lane line marker
            dict(
                type="line", y0=160, x0=80, y1=170, x1=80,
                line=dict(color=main_line_col, width=1), layer='below'
            ), # lane line marker
            dict(
                type="line", y0=160, x0=110, y1=170, x1=110,
                line=dict(color=main_line_col, width=1), layer='below'
            ), # lane line marker
            dict(
                type="line", y0=160, x0=140, y1=170, x1=140,
                line=dict(color=main_line_col, width=1), layer='below'
            ), # lane line marker
            dict(
                type="line", y0=340, x0=70, y1=330, x1=70,
                line=dict(color=main_line_col, width=1), layer='below'
            ), # lane line marker
            dict(
                type="line", y0=340, x0=80, y1=330, x1=80,
                line=dict(color=main_line_col, width=1), layer='below'
            ), # lane line marker
            dict(
                type="line", y0=340, x0=110, y1=330, x1=110,
                line=dict(color=main_line_col, width=1), layer='below'
            ), # lane line marker
            dict(
                type="line", y0=340, x0=140, y1=330, x1=140,
                line=dict(color=main_line_col, width=1), layer='below'
            ), # lane line marker

            ## right half

            dict(
                type="rect", y0=170, x0=750, y1=330, x1=940,
                line=dict(color=main_line_col, width=1),
                # fillcolor='#333333',
                layer='below'
            ),  # lane line rect, deltax=190
            dict(
                type="rect", y0=190, x0=750, y1=310, x1=940,
                line=dict(color=main_line_col, width=1),
                # fillcolor='#333333',
                layer='below'
            ),  # foul line rect, delta x=190
            dict(
                type="circle", y0=190, x0=690, y1=310, x1=810, xref="x", yref="y",
                line=dict(color=main_line_col, width=1),
                # fillcolor='#dddddd',
                layer='below'
            ),  # free-throw circle: r=60,c_x=190
            # dict(
            #     type="line", y0=190, x0=190, y1=310, x1=190,
            #     line=dict(color=main_line_col, width=1),
            #     layer='below'
            # ),  # foul line
            dict(
                type="rect", y0=248, x0=894.75, y1=252, x1=900,
                line=dict(color="#ec7607", width=1),
                fillcolor='#ec7607',
            ),  # hoop rect
            dict(
                type="circle", y0=242.5, x0=880, y1=257.5, x1=895, xref="x", yref="y",
                line=dict(color="#ec7607", width=1),
            ),  # hoop circle, d=15
            dict(
                type="line", y0=220, x0=900, y1=280, x1=900,
                line=dict(color="#ec7607", width=1),
            ),  # backboard

            dict(type="path",
                 path=ellipse_arc(a=40, b=40,
                                  start_angle=np.pi * 0.5, end_angle=np.pi * 1.5,
                                  x_center=887.5
                                  ),
                 line=dict(color=main_line_col, width=1), layer='below'),  # no-change semi-circle
            dict(type="path",
                 path=ellipse_arc(a=237.5, b=237.5, start_angle=0.386283101 +np.pi * 0.5,
                                  end_angle=np.pi * 1.5 - 0.386283101,
                                  x_center = 887.5
                                  ),
                 line=dict(color=main_line_col, width=1), layer='below'),  # three-point line:arc
            dict(
                type="line", y0=30, x0=887.5-threept_break_y, y1=30, x1=940,
                line=dict(color=three_line_col, width=1), layer='below'
            ),  # three-point line:left edge
            # dict(
            #     type="line", x0=-220, y0=-52.5, x1=-220, y1=threept_break_y,
            #     line=dict(color=three_line_col, width=1), layer='below'
            # ),
            dict(
                type="line", y0=470, x0=887.5-threept_break_y, y1=470, x1=940,
                line=dict(color=three_line_col, width=1), layer='below'
            ),  # three-point line:right edge

            dict(
                type="line", y0=0, x0=660, y1=30, x1=660,
                line=dict(color=main_line_col, width=1), layer='below'
            ),  # midcourt area marker:left
            dict(
                type="line", y0=520, x0=660, y1=470, x1=660,
                line=dict(color=main_line_col, width=1), layer='below'
            ),  # midcourt area marker:right
            dict(
                type="line", y0=160, x0=870, y1=170, x1=870,
                line=dict(color=main_line_col, width=1), layer='below'
            ),  # lane line marker
            dict(
                type="line", y0=160, x0=860, y1=170, x1=860,
                line=dict(color=main_line_col, width=1), layer='below'
            ),  # lane line marker
            dict(
                type="line", y0=160, x0=830, y1=170, x1=830,
                line=dict(color=main_line_col, width=1), layer='below'
            ),  # lane line marker
            dict(
                type="line", y0=160, x0=800, y1=170, x1=800,
                line=dict(color=main_line_col, width=1), layer='below'
            ),  # lane line marker
            dict(
                type="line", y0=340, x0=870, y1=330, x1=870,
                line=dict(color=main_line_col, width=1), layer='below'
            ),  # lane line marker
            dict(
                type="line", y0=340, x0=860, y1=330, x1=860,
                line=dict(color=main_line_col, width=1), layer='below'
            ),  # lane line marker
            dict(
                type="line", y0=340, x0=830, y1=330, x1=830,
                line=dict(color=main_line_col, width=1), layer='below'
            ),  # lane line marker
            dict(
                type="line", y0=340, x0=800, y1=330, x1=800,
                line=dict(color=main_line_col, width=1), layer='below'
            ),  # lane line marker

        ]
    )
    return True


model_dir='/media/felicia/Data/sgan_results/best_models'
models=[
    'sm.baseline_attention_v3.6.d5.e16.dg10.gg10_with_model.pt',
    'sm.team_pos_attentiontp_v3.6.d5.e16.pe16.te4.tpd5.gg10.dg10.l10_with_model.pt',
    'cs05.baseline_attention_v3.6.d5.e16.dg10.gg10_with_model.pt',
    'cs05.team_pos_attentiontp_v3.6.d0.e16.pe16.te4.tpd5.gg8.dg8.l10_with_model.pt'
]

midx=0
model_name=models[midx]
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


## fig axis: 940->28
# x=24
# factor=0.3048 # foot to meet
# new_x=x/factor/94*940

factor=10/0.3048

colors_set2=px.colors.qualitative.Set2
colors_pastel2=px.colors.qualitative.Pastel2
colors=[colors_set2[0],colors_pastel2[0],colors_set2[2],colors_pastel2[2],colors_set2[1],colors_pastel2[1],] # green,blue, orange
"""

"""

Teams=["Team 0","Team 1", "Ball"]

# sample_idx={0: [0,1500,2500],
#             1: [0, 1500,2500],
#             2: [0, 1500, 2000,3000, 5000,5500, 6500],
#             3: [0, 1500, 2000,3000, 5000,5500, 6500]}

sample_idx={0: [0,],
            1: [0,],
            2: [ 2000],
            3: [2000]}

# for i in tqdm(range(ntraj)):
#     if i%500==0:

for i in sample_idx[midx]:
    fig_gt= go.Figure()
    draw_plotly_whole_court(fig_gt)
    legend_set_gt = set()

    fig_pred = go.Figure()
    draw_plotly_whole_court(fig_pred)
    legend_set_pred = set()

    for j in range(11):
        obs_x,obs_y=obs_traj_np[:,i,j,0],obs_traj_np[:,i,j,1] # 8
        pred_x,pred_y=pred_traj_fake_np[:,i,j,0],pred_traj_fake_np[:,i,j,1] # 8
        gt_x,gt_y=pred_traj_gt_np[:,i,j,0],pred_traj_gt_np[:,i,j,1] # 8

        obs_x,obs_y=obs_x*factor,obs_y*factor
        pred_x,pred_y=pred_x*factor,pred_y*factor
        gt_x,gt_y=gt_x*factor,gt_y*factor


        _,team_id,color_obs,color_pred=vector_color(None,team_vec_np[:,i,j])
        color_obs_=[colors[c] for c in color_obs]
        color_pred_=[colors[c] for c in color_pred]

        if Teams[team_id[0]] not in legend_set_gt:
            legend_set_gt.add(Teams[team_id[0]])
            showlegend_gt=True
        else:
            showlegend_gt=False

        fig_gt.add_trace(go.Scatter(x=obs_x, y=obs_y, mode='markers', showlegend=showlegend_gt, name=Teams[team_id[0]]+' Observation',
                                 marker=dict(color=color_obs_,size=7,
                                    line=dict(width=.5,color='DarkSlateGrey'))))
        fig_gt.add_trace(go.Scatter(x=gt_x, y=gt_y, mode='markers',showlegend=showlegend_gt,  name=Teams[team_id[0]]+' Ground Truth',
                                 marker=dict(color=color_pred_,size=7,
                                     line=dict(color='DarkSlateGrey',width=.5))))

        if Teams[team_id[0]] not in legend_set_pred:
            legend_set_pred.add(Teams[team_id[0]])
            showlegend_pred=True
        else:
            showlegend_pred=False

        fig_pred.add_trace(go.Scatter(x=obs_x, y=obs_y, mode='markers', showlegend=showlegend_pred, name=Teams[team_id[0]]+' Observation',
                                 marker=dict(color=color_obs_,size=7,
                                    line=dict(width=.5,color='DarkSlateGrey'))))
        fig_pred.add_trace(go.Scatter(x=pred_x, y=pred_y, mode='markers',showlegend=showlegend_pred,  name=Teams[team_id[0]]+' Prediction',
                                 marker=dict(color=color_pred_,size=7,
                                     line=dict(color='DarkSlateGrey',width=.5))))

    # vis_dir = '/media/felicia/Data/sgan_results/vis/traj/'
    # fig_gt.write_image(vis_dir + "court_model{:01d}_traj{:04d}_gt.png".format(midx,i))
    #
    # vis_dir = '/media/felicia/Data/sgan_results/vis/traj/'
    # fig_pred.write_image(vis_dir + "court_model{:01d}_traj{:04d}_pred.png".format(midx,i))

    vis_dir = '/media/felicia/Data/sgan_results/vis/pdf/'
    fig_gt.update_layout(
        legend=dict(font=dict( size=22,color="black")),
    )
    fig_gt.write_image(vis_dir + "court_model{:01d}_traj{:04d}_gt.pdf".format(midx,i))

    vis_dir = '/media/felicia/Data/sgan_results/vis/pdf/'
    fig_pred.update_layout(
        legend=dict(font=dict( size=22,color="black")),
    )
    fig_pred.write_image(vis_dir + "court_model{:01d}_traj{:04d}_pred.pdf".format(midx,i))
    break


# fig.show()

"""
model 0: 0,1500,2500
model 1: 0, 1500,  
model 2: 0, 1500, 2000,3000, 5000, 6500
model 3: 1500, 2000, 5500, 6500
"""
