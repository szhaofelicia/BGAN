import numpy as np
# import plotly
import plotly.graph_objects as go



def draw_plotly_half_court(fig, fig_width=600, margins=10):
    # From: https://community.plot.ly/t/arc-shape-with-path/7205/5
    def ellipse_arc(x_center=0.0, y_center=0.0, a=10.5, b=10.5, start_angle=0.0, end_angle=2 * np.pi, N=200, closed=False):
        t = np.linspace(start_angle, end_angle, N)
        x = x_center + a * np.cos(t)
        y = y_center + b * np.sin(t)
        path = f'M {x[0]}, {y[0]}'
        for k in range(1, len(t)):
            path += f'L{x[k]}, {y[k]}'
        if closed:
            path += ' Z'
        return path

    fig_height = fig_width * (470 + 2 * margins) / (500 + 2 * margins)
    fig.update_layout(width=fig_width, height=fig_height)

    # Set axes ranges
    fig.update_xaxes(range=[-250 - margins, 250 + margins])
    fig.update_yaxes(range=[-52.5 - margins, 417.5 + margins])

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

        shapes=[
        # half_layout=[
            dict(
                type="rect", x0=-250, y0=-52.5, x1=250, y1=417.5,
                line=dict(color=main_line_col, width=1),
                # fillcolor='#333333',
                layer='below'
            ), ## sideline rect
            dict(
                type="rect", x0=-80, y0=-52.5, x1=80, y1=137.5,
                line=dict(color=main_line_col, width=1),
                # fillcolor='#333333',
                layer='below'
            ),# lane line rect
            dict(
                type="rect", x0=-60, y0=-52.5, x1=60, y1=137.5,
                line=dict(color=main_line_col, width=1),
                # fillcolor='#333333',
                layer='below'
            ), # foul line rect
            dict(
                type="circle", x0=-60, y0=77.5, x1=60, y1=197.5, xref="x", yref="y",
                line=dict(color=main_line_col, width=1),
                # fillcolor='#dddddd',
                layer='below'
            ), # free-throw circle
            dict(
                type="line", x0=-60, y0=137.5, x1=60, y1=137.5,
                line=dict(color=main_line_col, width=1),
                layer='below'
            ), # foul line

            dict(
                type="rect", x0=-2, y0=-7.25, x1=2, y1=-12.5,
                line=dict(color="#ec7607", width=1),
                fillcolor='#ec7607',
            ), # hoop rect
            dict(
                type="circle", x0=-7.5, y0=-7.5, x1=7.5, y1=7.5, xref="x", yref="y",
                line=dict(color="#ec7607", width=1),
            ), # hoop circle
            dict(
                type="line", x0=-30, y0=-12.5, x1=30, y1=-12.5,
                line=dict(color="#ec7607", width=1),
            ), # backboard

            dict(type="path",
                 path=ellipse_arc(a=40, b=40, start_angle=0, end_angle=np.pi),
                 line=dict(color=main_line_col, width=1), layer='below'), # no-change semi-circle
            dict(type="path",
                 path=ellipse_arc(a=237.5, b=237.5, start_angle=0.386283101, end_angle=np.pi - 0.386283101),
                 line=dict(color=main_line_col, width=1), layer='below'), # three-point line:arc
            dict(
                type="line", x0=-220, y0=-52.5, x1=-220, y1=threept_break_y,
                line=dict(color=three_line_col, width=1), layer='below'
            ), # three-point line:left edge
            # dict(
            #     type="line", x0=-220, y0=-52.5, x1=-220, y1=threept_break_y,
            #     line=dict(color=three_line_col, width=1), layer='below'
            # ),
            dict(
                type="line", x0=220, y0=-52.5, x1=220, y1=threept_break_y,
                line=dict(color=three_line_col, width=1), layer='below'
            ), # three-point line:right edge

            dict(
                type="line", x0=-250, y0=227.5, x1=-220, y1=227.5,
                line=dict(color=main_line_col, width=1), layer='below'
            ), # midcourt area marker:left
            dict(
                type="line", x0=250, y0=227.5, x1=220, y1=227.5,
                line=dict(color=main_line_col, width=1), layer='below'
            ), # midcourt area marker:right
            dict(
                type="line", x0=-90, y0=17.5, x1=-80, y1=17.5,
                line=dict(color=main_line_col, width=1), layer='below'
            ), # lane line marker
            dict(
                type="line", x0=-90, y0=27.5, x1=-80, y1=27.5,
                line=dict(color=main_line_col, width=1), layer='below'
            ), # lane line marker
            dict(
                type="line", x0=-90, y0=57.5, x1=-80, y1=57.5,
                line=dict(color=main_line_col, width=1), layer='below'
            ), # lane line marker
            dict(
                type="line", x0=-90, y0=87.5, x1=-80, y1=87.5,
                line=dict(color=main_line_col, width=1), layer='below'
            ), # lane line marker
            dict(
                type="line", x0=90, y0=17.5, x1=80, y1=17.5,
                line=dict(color=main_line_col, width=1), layer='below'
            ), # lane line marker
            dict(
                type="line", x0=90, y0=27.5, x1=80, y1=27.5,
                line=dict(color=main_line_col, width=1), layer='below'
            ), # lane line marker
            dict(
                type="line", x0=90, y0=57.5, x1=80, y1=57.5,
                line=dict(color=main_line_col, width=1), layer='below'
            ), # lane line marker
            dict(
                type="line", x0=90, y0=87.5, x1=80, y1=87.5,
                line=dict(color=main_line_col, width=1), layer='below'
            ), # lane line marker

            dict(type="path",
                 path=ellipse_arc(y_center=417.5, a=60, b=60, start_angle=-0, end_angle=-np.pi),
                 line=dict(color=main_line_col, width=1), layer='below'), # center circle: half

        ]
    )
    return True

def draw_plotly_whole_court(fig, fig_width=600, margins=10):
    # From: https://community.plot.ly/t/arc-shape-with-path/7205/5
    def ellipse_arc(x_center=0.0, y_center=0.0, a=10.5, b=10.5, start_angle=0.0, end_angle=2 * np.pi, N=200, closed=False):
        t = np.linspace(start_angle, end_angle, N)
        x = x_center + a * np.cos(t)
        y = y_center + b * np.sin(t)
        path = f'M {x[0]}, {y[0]}'
        for k in range(1, len(t)):
            path += f'L{x[k]}, {y[k]}'
        if closed:
            path += ' Z'
        return path

    fig_height = fig_width * (470*2 + 2 * margins) / (500 + 2 * margins)
    fig.update_layout(width=fig_width, height=fig_height)

    # Set axes ranges
    fig.update_xaxes(range=[-250 - margins, 250 + margins])
    fig.update_yaxes(range=[-52.5 - margins, 417.5+470 + margins])

    # fig.update_xaxes(range=[ margins, 500 + margins])
    # fig.update_yaxes(range=[margins, 470*2 + margins])

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
        shapes=[
            dict(
                type="rect", x0=-250, y0=-52.5, x1=250, y1=417.5+470,
                line=dict(color=main_line_col, width=1),
                # fillcolor='#333333',
                layer='below'
            ),  ## sideline rect
            # dict(
            #     type="rect", x0=-250, y0=-52.5, x1=250, y1=417.5,
            #     line=dict(color=main_line_col, width=1),
            #     # fillcolor='#333333',
            #     layer='below'
            # ), ## sideline rect
            dict(
                type="rect", x0=-80, y0=-52.5, x1=80, y1=137.5,
                line=dict(color=main_line_col, width=1),
                # fillcolor='#333333',
                layer='below'
            ),# lane line rect
            dict(
                type="rect", x0=-60, y0=-52.5, x1=60, y1=137.5,
                line=dict(color=main_line_col, width=1),
                # fillcolor='#333333',
                layer='below'
            ), # foul line rect
            dict(
                type="circle", x0=-60, y0=77.5, x1=60, y1=197.5, xref="x", yref="y",
                line=dict(color=main_line_col, width=1),
                # fillcolor='#dddddd',
                layer='below'
            ), # free-throw circle
            dict(
                type="line", x0=-60, y0=137.5, x1=60, y1=137.5,
                line=dict(color=main_line_col, width=1),
                layer='below'
            ), # foul line

            dict(
                type="rect", x0=-2, y0=-7.25, x1=2, y1=-12.5,
                line=dict(color="#ec7607", width=1),
                fillcolor='#ec7607',
            ), # hoop rect
            dict(
                type="circle", x0=-7.5, y0=-7.5, x1=7.5, y1=7.5, xref="x", yref="y",
                line=dict(color="#ec7607", width=1),
            ), # hoop circle
            dict(
                type="line", x0=-30, y0=-12.5, x1=30, y1=-12.5,
                line=dict(color="#ec7607", width=1),
            ), # backboard

            dict(type="path",
                 path=ellipse_arc(a=40, b=40, start_angle=0, end_angle=np.pi),
                 line=dict(color=main_line_col, width=1), layer='below'), # no-change semi-circle
            dict(type="path",
                 path=ellipse_arc(a=237.5, b=237.5, start_angle=0.386283101, end_angle=np.pi - 0.386283101),
                 line=dict(color=main_line_col, width=1), layer='below'), # three-point line:arc
            dict(
                type="line", x0=-220, y0=-52.5, x1=-220, y1=threept_break_y,
                line=dict(color=three_line_col, width=1), layer='below'
            ), # three-point line:left edge
            # dict(
            #     type="line", x0=-220, y0=-52.5, x1=-220, y1=threept_break_y,
            #     line=dict(color=three_line_col, width=1), layer='below'
            # ),
            dict(
                type="line", x0=220, y0=-52.5, x1=220, y1=threept_break_y,
                line=dict(color=three_line_col, width=1), layer='below'
            ), # three-point line:right edge

            dict(
                type="line", x0=-250, y0=227.5, x1=-220, y1=227.5,
                line=dict(color=main_line_col, width=1), layer='below'
            ), # midcourt area marker:left
            dict(
                type="line", x0=250, y0=227.5, x1=220, y1=227.5,
                line=dict(color=main_line_col, width=1), layer='below'
            ), # midcourt area marker:right
            dict(
                type="line", x0=-90, y0=17.5, x1=-80, y1=17.5,
                line=dict(color=main_line_col, width=1), layer='below'
            ), # lane line marker
            dict(
                type="line", x0=-90, y0=27.5, x1=-80, y1=27.5,
                line=dict(color=main_line_col, width=1), layer='below'
            ), # lane line marker
            dict(
                type="line", x0=-90, y0=57.5, x1=-80, y1=57.5,
                line=dict(color=main_line_col, width=1), layer='below'
            ), # lane line marker
            dict(
                type="line", x0=-90, y0=87.5, x1=-80, y1=87.5,
                line=dict(color=main_line_col, width=1), layer='below'
            ), # lane line marker
            dict(
                type="line", x0=90, y0=17.5, x1=80, y1=17.5,
                line=dict(color=main_line_col, width=1), layer='below'
            ), # lane line marker
            dict(
                type="line", x0=90, y0=27.5, x1=80, y1=27.5,
                line=dict(color=main_line_col, width=1), layer='below'
            ), # lane line marker
            dict(
                type="line", x0=90, y0=57.5, x1=80, y1=57.5,
                line=dict(color=main_line_col, width=1), layer='below'
            ), # lane line marker
            dict(
                type="line", x0=90, y0=87.5, x1=80, y1=87.5,
                line=dict(color=main_line_col, width=1), layer='below'
            ), # lane line marker

            dict(type="path",
                 path=ellipse_arc(y_center=417.5, a=60, b=60, start_angle=-0, end_angle=-np.pi),
                 line=dict(color=main_line_col, width=1), layer='below'), # center circle: half


            ## upper
            # dict(
            #     type="rect", x0=-250, y0=-52.5, x1=250, y1=417.5,
            #     line=dict(color=main_line_col, width=1),
            #     # fillcolor='#333333',
            #     layer='below'
            # ),  ## sideline rect
            # dict(
            #     type="rect", x0=-80, y0=-52.5, x1=80, y1=137.5,
            #     line=dict(color=main_line_col, width=1),
            #     # fillcolor='#333333',
            #     layer='below'
            # ),  # lane line rect
            # dict(
            #     type="rect", x0=-60, y0=-52.5, x1=60, y1=137.5,
            #     line=dict(color=main_line_col, width=1),
            #     # fillcolor='#333333',
            #     layer='below'
            # ),  # foul line rect
            # dict(
            #     type="circle", x0=-60, y0=77.5, x1=60, y1=197.5, xref="x", yref="y",
            #     line=dict(color=main_line_col, width=1),
            #     # fillcolor='#dddddd',
            #     layer='below'
            # ),  # free-throw circle
            # dict(
            #     type="line", x0=-60, y0=137.5, x1=60, y1=137.5,
            #     line=dict(color=main_line_col, width=1),
            #     layer='below'
            # ),  # foul line
            #
            # dict(
            #     type="rect", x0=-2, y0=-7.25, x1=2, y1=-12.5,
            #     line=dict(color="#ec7607", width=1),
            #     fillcolor='#ec7607',
            # ),  # hoop rect
            # dict(
            #     type="circle", x0=-7.5, y0=-7.5, x1=7.5, y1=7.5, xref="x", yref="y",
            #     line=dict(color="#ec7607", width=1),
            # ),  # hoop circle
            # dict(
            #     type="line", x0=-30, y0=-12.5, x1=30, y1=-12.5,
            #     line=dict(color="#ec7607", width=1),
            # ),  # backboard
            #
            # dict(type="path",
            #      path=ellipse_arc(a=40, b=40, start_angle=0, end_angle=np.pi),
            #      line=dict(color=main_line_col, width=1), layer='below'),  # no-change semi-circle
            # dict(type="path",
            #      path=ellipse_arc(a=237.5, b=237.5, start_angle=0.386283101, end_angle=np.pi - 0.386283101),
            #      line=dict(color=main_line_col, width=1), layer='below'),  # three-point line:arc
            # dict(
            #     type="line", x0=-220, y0=-52.5, x1=-220, y1=threept_break_y,
            #     line=dict(color=three_line_col, width=1), layer='below'
            # ),  # three-point line:left edge
            # # dict(
            # #     type="line", x0=-220, y0=-52.5, x1=-220, y1=threept_break_y,
            # #     line=dict(color=three_line_col, width=1), layer='below'
            # # ),
            # dict(
            #     type="line", x0=220, y0=-52.5, x1=220, y1=threept_break_y,
            #     line=dict(color=three_line_col, width=1), layer='below'
            # ),  # three-point line:right edge
            #
            # dict(
            #     type="line", x0=-250, y0=227.5, x1=-220, y1=227.5,
            #     line=dict(color=main_line_col, width=1), layer='below'
            # ),  # midcourt area marker:left
            # dict(
            #     type="line", x0=250, y0=227.5, x1=220, y1=227.5,
            #     line=dict(color=main_line_col, width=1), layer='below'
            # ),  # midcourt area marker:right
            # dict(
            #     type="line", x0=-90, y0=17.5, x1=-80, y1=17.5,
            #     line=dict(color=main_line_col, width=1), layer='below'
            # ),  # lane line marker
            # dict(
            #     type="line", x0=-90, y0=27.5, x1=-80, y1=27.5,
            #     line=dict(color=main_line_col, width=1), layer='below'
            # ),  # lane line marker
            # dict(
            #     type="line", x0=-90, y0=57.5, x1=-80, y1=57.5,
            #     line=dict(color=main_line_col, width=1), layer='below'
            # ),  # lane line marker
            # dict(
            #     type="line", x0=-90, y0=87.5, x1=-80, y1=87.5,
            #     line=dict(color=main_line_col, width=1), layer='below'
            # ),  # lane line marker
            # dict(
            #     type="line", x0=90, y0=17.5, x1=80, y1=17.5,
            #     line=dict(color=main_line_col, width=1), layer='below'
            # ),  # lane line marker
            # dict(
            #     type="line", x0=90, y0=27.5, x1=80, y1=27.5,
            #     line=dict(color=main_line_col, width=1), layer='below'
            # ),  # lane line marker
            # dict(
            #     type="line", x0=90, y0=57.5, x1=80, y1=57.5,
            #     line=dict(color=main_line_col, width=1), layer='below'
            # ),  # lane line marker
            # dict(
            #     type="line", x0=90, y0=87.5, x1=80, y1=87.5,
            #     line=dict(color=main_line_col, width=1), layer='below'
            # ),  # lane line marker
            #
            # dict(type="path",
            #      path=ellipse_arc(y_center=417.5, a=60, b=60, start_angle=-0, end_angle=-np.pi),
            #      line=dict(color=main_line_col, width=1), layer='below'),  # center circle: half

        ]
    )
    return True






max_freq = 0.002
# freq_by_hex = np.array([min(max_freq, i) for i in league_hexbin_stats['freq_by_hex']])
colorscale = 'YlOrRd'
marker_cmin = 0.1
marker_cmax = 0.6
ticktexts = [str(marker_cmin*100)+'%-', "", str(marker_cmax*100)+'%+']

fig = go.Figure()
# draw_plotly_half_court(fig)
draw_plotly_whole_court(fig)

# fig.add_trace(go.Scatter(
#     x=xlocs, y=ylocs, mode='markers', name='markers',
#     marker=dict(
#         size=freq_by_hex, sizemode='area', sizeref=2. * max(freq_by_hex) / (11. ** 2), sizemin=2.5,
#         color=accs_by_hex, colorscale=colorscale,
#         colorbar=dict(
#             thickness=15,
#             x=0.84,
#             y=0.87,
#             yanchor='middle',
#             len=0.2,
#             title=dict(
#                 text="<B>Accuracy</B>",
#                 font=dict(
#                     size=11,
#                     color='#4d4d4d'
#                 ),
#             ),
#             tickvals=[marker_cmin, (marker_cmin + marker_cmax) / 2, marker_cmax],
#             ticktext=ticktexts,
#             tickfont=dict(
#                 size=11,
#                 color='#4d4d4d'
#             )
#         ),
#         cmin=marker_cmin, cmax=marker_cmax,
#         line=dict(width=1, color='#333333'), symbol='hexagon',
#     ),
# ))
# fig.show(config=dict(displayModeBar=False))

# fig.show()

vis_dir='/media/felicia/Data/sgan_results/vis/'
fig.write_image(vis_dir+"court.svg")
