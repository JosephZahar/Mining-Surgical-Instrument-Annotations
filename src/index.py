# import desired libraries
import itertools
import pathlib
from collections import deque
import pandas as pd
import plotly.io as pio
from sklearn.cluster import KMeans

pio.templates.default = "simple_white"
import plotly.offline as pyo

# pyo.init_notebook_mode()
import plotly.graph_objects as go
import plotly.express as px
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import dash
import ast
import statsmodels.api as sm
import random
import warnings
import statistics
import numpy as np

warnings.filterwarnings("ignore")


def flatten_r(L):
    L1 = deque(L)
    L1.appendleft([0])
    L1 = list(L1)
    return list(itertools.chain.from_iterable(L1))


PATH = pathlib.Path(__file__).parent.parent
DATA_PATH = PATH.joinpath("data").resolve()
df = pd.read_csv(DATA_PATH.joinpath('rp_proportional_durations.csv')) \
    .loc[:, ['video_uuid', 'surgeon_uuid', 'Account', 'content', 'content_type', 'start', 'end']]

df_tool = df[df['content_type'] == 'tooling'].drop_duplicates(subset=None, keep="first", inplace=False)
df_phase = df[df['content_type'] == 'phase'].drop_duplicates(subset=None, keep="first", inplace=False)
df_phase = df_phase.merge(right=df_tool['video_uuid'], how='inner', on='video_uuid')
df_phase = df_phase.drop_duplicates(subset=None, keep="first", inplace=False)

df_phase['start_end'] = df_phase.loc[:, ['start', 'end']].values.tolist()
grouped_phase = df_phase.groupby(['video_uuid', 'content'])['start_end'].apply(list).reset_index()
max_time = df_phase.groupby('video_uuid')['end'].max()
grouped_phase = grouped_phase.merge(right=max_time, how='left', on='video_uuid')
grouped_phase['start_end'] = grouped_phase['start_end'].apply(lambda x: flatten_r(x))
grouped_phase['start_end2'] = grouped_phase.apply(lambda row: row['start_end'].append(row['end']), axis=1)
grouped_phase['diff'] = grouped_phase['start_end'].apply(lambda x: np.diff(x))
grouped_phase = grouped_phase[grouped_phase['content'] != 'Operation Finished']

grouped_phase_content = grouped_phase.groupby('video_uuid')['content'].apply(list).reset_index()
grouped_phase_time = grouped_phase.groupby('video_uuid')['start_end'].apply(list).reset_index().drop('video_uuid',
                                                                                                     axis=1)
grouped_phase_list = pd.concat([grouped_phase_content, grouped_phase_time], axis=1)
phases_list = grouped_phase_list.content.values
phases_times = grouped_phase_list.start_end.values
grouped_phase_list = pd.merge(grouped_phase_list, df_phase.groupby('video_uuid')['Account'].first(), how='inner',
                              on='video_uuid')
accounts = grouped_phase_list.Account.values

timeseries_compressed_df = pd.read_csv(DATA_PATH.joinpath('timeseries_compressed_df.csv'))
timeseries_compressed_df.seq = timeseries_compressed_df.seq.apply(lambda row: ast.literal_eval(row))
timeseries_compressed = timeseries_compressed_df.seq.values

timeseries_compressed_100k_df = pd.read_csv(DATA_PATH.joinpath('timeseries_compressed_100k_df.csv'))
timeseries_compressed_100k_df.seq = timeseries_compressed_100k_df.seq.apply(lambda row: ast.literal_eval(row))
timeseries_compressed_100k = timeseries_compressed_100k_df.seq.values

timeseries_compressed_10k_df = pd.read_csv(DATA_PATH.joinpath('timeseries_compressed_10k_df.csv'))
timeseries_compressed_10k_df.seq = timeseries_compressed_10k_df.seq.apply(lambda row: ast.literal_eval(row))
timeseries_compressed_10k = timeseries_compressed_10k_df.seq.values


def autocorr_dash(seq, surgery_number):
    surgery_number = int(surgery_number[10:])-1
    if seq == 'Low Compression':
        timeseries = timeseries_compressed_10k
        reduct = 10000
    elif seq == 'Medium Compression':
        timeseries = timeseries_compressed
        reduct = 50000
    else:
        timeseries = timeseries_compressed_100k
        reduct = 100000

    lags = 30
    acf_list = []
    for i in timeseries:
        acf_list.append(sm.tsa.stattools.acf(i, nlags=lags))

    fig1 = go.Figure()
    for i in range(len(acf_list)):
        if i == surgery_number:
            fig1.add_trace(go.Scatter(x=[j for j in range(lags + 1)], y=acf_list[i], mode='lines',
                                      marker_color=('orange'), line=dict(width=4)))
        else:
            fig1.add_trace(
                go.Scatter(x=[j for j in range(lags + 1)], y=acf_list[i], mode='lines', marker_color=('gray'),
                           line=dict(width=0.5)))
    fig1.update_layout(xaxis_title='Lag', yaxis_title='Autocorrelation', title='Autocorrelation across different lags')
    fig1.update_layout(showlegend=False)

    low_thresh = []
    for i in range(len(timeseries)):
        if acf_list[i][1] < 0.5:
            low_thresh.append(i)

    high_thresh = []
    for i in range(len(timeseries)):
        if acf_list[i][1] > 0.5:
            high_thresh.append(i)

    def inst_change_dec(seq):
        L = []
        for i in range(len(seq) - 2):
            if seq[i] > seq[i + 1]:
                if (seq[i] != seq[i + 1]) and (seq[i + 1] != seq[i + 2]) and (seq[i + 1] < seq[i + 2]):
                    L = L + [i + 1, i + 2]
            elif seq[i] < seq[i + 1]:
                if (seq[i] != seq[i + 1]) and (seq[i + 1] != seq[i + 2]) and (seq[i + 1] > seq[i + 2]):
                    L = L + [i + 1, i + 2]
        nums = sorted(set(L))
        gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s + 1 < e]
        edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
        ranges = list(zip(edges, edges))
        return ranges

    def inst_change_inc(seq):
        L = []
        for i in range(len(seq) - 2):
            if seq[i] > seq[i + 1]:
                if (seq[i] != seq[i + 1]) and (seq[i + 1] != seq[i + 2]) and (seq[i + 1] > seq[i + 2]):
                    L = L + [i + 1, i + 2]
            elif seq[i] < seq[i + 1]:
                if (seq[i] != seq[i + 1]) and (seq[i + 1] != seq[i + 2]) and (seq[i + 1] < seq[i + 2]):
                    L = L + [i + 1, i + 2]
        nums = sorted(set(L))
        gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s + 1 < e]
        edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
        ranges = list(zip(edges, edges))
        return ranges

    timeseries_lbl = []
    for i in range(len(timeseries[int(surgery_number)])):
        if timeseries[surgery_number][i] == 1:
            timeseries_lbl.append('Instrument 1')
        elif timeseries[surgery_number][i] == 2:
            timeseries_lbl.append('Instrument 2')
        elif timeseries[surgery_number][i] == 3:
            timeseries_lbl.append('Both instruments')
        else:
            timeseries_lbl.append('None')

    ranges_dec = inst_change_dec(timeseries[surgery_number])
    ranges_inc = inst_change_inc(timeseries[surgery_number])
    fig2 = px.line(x=[i for i in range(len(timeseries_lbl))], y=timeseries_lbl, markers=False)
    fig2.update_traces(line_color='#005AB5')
    for i in ranges_dec:
        fig2.add_trace(go.Scatter(x=[i for i in range(i[0], i[1] + 1)], y=timeseries_lbl[i[0]:i[1] + 1], mode='lines',
                                  marker_color=('#4B0092'), line=dict(width=5)))
    for i in ranges_inc:
        fig2.add_trace(go.Scatter(x=[i for i in range(i[0], i[1] + 1)], y=timeseries_lbl[i[0]:i[1] + 1], mode='lines',
                                  marker_color=('#1AFF1A'), line=dict(width=5)))
    fig2.update_layout(showlegend=False)
    fig2.update_layout(xaxis_title="Time", yaxis_title="Instrument", title='Tool Switches Across Time')
    fig2.update_yaxes(categoryorder='array', categoryarray=['None', 'Instrument 1', 'Instrument 2', 'Both instruments'])

    phasess = grouped_phase.groupby('content')['video_uuid'].first().reset_index()['content'].values
    phase_color_dict = dict()
    random.seed(42)
    colors = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
              for i in range(len(phasess) + 20)]
    for phase, color in zip(phasess, colors):
        phase_color_dict[phase] = color
    fig3 = px.line(x=[i for i in range(len(timeseries_lbl))], y=timeseries_lbl, markers=False)
    for i, value in enumerate(phases_times[surgery_number]):
        phase_color = phase_color_dict[phasess[i]]
        for j in range(1, len(value) - 1, 2):
            fig3.add_shape(type="rect",
                           x0=round(value[j] / reduct), y0=0, x1=round(value[j + 1] / reduct), y1=3.1,
                           line=dict(
                               color="black",
                               width=0.5,
                           ), opacity=0.4,
                           fillcolor=phase_color)
    fig3.update_layout(showlegend=False)
    fig3.update_traces(line_color='black')
    fig3.update_layout(xaxis_title="Time", yaxis_title="Instrument")
    fig3.update_yaxes(categoryorder='array', categoryarray=['None', 'Instrument 1', 'Instrument 2', 'Both instruments'])

    acf_list_5lag = list(map(lambda l: l[:5], acf_list))
    kmeans_model = KMeans(n_clusters=2, random_state=0).fit(pd.DataFrame(acf_list_5lag))

    cluster1_indexes = []
    cluster2_indexes = []
    cluster1_trueindexes = []
    cluster2_trueindexes = []
    for j in range(len(acf_list)):
        if kmeans_model.labels_[j] == 1:
            cluster1_indexes.append(acf_list[j])
            cluster1_trueindexes.append(j)
        else:
            cluster2_indexes.append(acf_list[j])
            cluster2_trueindexes.append(j)

    if statistics.mean([i[1] for i in cluster1_indexes]) > statistics.mean([i[1] for i in cluster2_indexes]):
        more = statistics.mean([i[1] for i in cluster1_indexes])
        less = statistics.mean([i[1] for i in cluster2_indexes])
    else:
        less = statistics.mean([i[1] for i in cluster1_indexes])
        more = statistics.mean([i[1] for i in cluster2_indexes])

    fig4 = go.Figure(data=[go.Bar(
        x=['Cluster 1', 'Cluster 2'],
        y=[less, more], text=[round(less,3), round(more,3)],
        marker_color=['#DC3220', '#005AB5'])])
    fig4.update_layout(uniformtext_minsize=20, uniformtext_mode='hide')
    fig4.update_layout(title='Average autocorrelation value at lag of 1')
    fig4.update_traces(width=0.5, textposition='outside', cliponaxis=False)

    def rep_count(tup):
        return tup[1] - tup[0]

    cluster1_tm = []
    cluster2_tm = []
    for j in range(len(timeseries)):
        if kmeans_model.labels_[j] == 1:
            cluster1_tm.append(timeseries[j])
        else:
            cluster2_tm.append(timeseries[j])

    range1_dec = [sum(list(map(rep_count, inst_change_dec(i))))*72 / len(i) for i in cluster1_tm]
    range2_dec = [sum(list(map(rep_count, inst_change_dec(i))))*72 / len(i) for i in cluster2_tm]
    range1_inc = [sum(list(map(rep_count, inst_change_inc(i))))*72 / len(i) for i in cluster1_tm]
    range2_inc = [sum(list(map(rep_count, inst_change_inc(i))))*72 / len(i) for i in cluster2_tm]
    range1 = [abs(range1_dec[x] - range1_inc[x]) for x in range(len(range1_dec))]
    range2 = [abs(range2_dec[x] - range2_inc[x]) for x in range(len(range2_dec))]

    if sum(range1) > sum(range2):
        index1 = 0
        index2 = 1
        tot_more = range1
        tot_less = range2
        more_indexes = cluster1_trueindexes
        less_indexes = cluster2_trueindexes
    else:
        index1 = 1
        index2 = 0
        tot_less = range1
        tot_more = range2
        more_indexes = cluster2_trueindexes
        less_indexes = cluster1_trueindexes

    fig5 = go.Figure()
    fig5.add_trace(go.Box(x=tot_less, name='Cluster 1',
                          marker_color='#DC3220', boxpoints='all', line={"color":"black"}, fillcolor='#DC3220',
                          text=['index {}'.format(less_indexes[i]) for i in range(len(less_indexes))]))
    fig5.add_trace(go.Box(x=tot_more, name='Cluster 2',
                          marker_color='#005AB5', boxpoints='all', line={"color":"black"}, fillcolor='#005AB5',
                          text=['index {}'.format(more_indexes[i]) for i in range(len(tot_more))]))
    fig5.update_layout(title='Distribution of Instrument Swaps/h')
    fig5.update_layout(showlegend=False)
    fig5.update_layout(hovermode='y unified')

    fig6 = go.Figure()
    for i in range(len(acf_list)):
        if kmeans_model.labels_[i] == index2:
            fig6.add_trace(go.Scatter(x=[j for j in range(lags + 1)], y=acf_list[i], mode='lines',
                                      marker_color=('#005AB5'), line=dict(width=2), name='Cluster 2'))
        else:
            fig6.add_trace(
                go.Scatter(x=[j for j in range(lags + 1)], y=acf_list[i], mode='lines', marker_color=('#DC3220'),
                           name='Cluster 1',
                           line=dict(width=2)))
    fig6.update_layout(xaxis_title='Lag', yaxis_title='Autocorrelation', title='Autocorrelation Clusters')
    fig6.update_layout(showlegend=False)

    df_plot_shift = pd.DataFrame()
    for i in range(lags):
        if i == 0:
            lag_tm = timeseries_lbl
        else:
            lag_tm = [None] * i + timeseries_lbl[:-i]
        to_concat = pd.DataFrame(list(zip([i for i in range(len(timeseries_lbl))], lag_tm, [i] * len(timeseries_lbl))),
                                 columns=['time', 'autocorr', 'Lag'])
        df_plot_shift = pd.concat([df_plot_shift, to_concat])

    fig7 = px.line(df_plot_shift, x='time', y='autocorr', animation_frame="Lag", markers=False)
    fig7["layout"].pop("updatemenus")
    fig7.update_traces(line_color='#005AB5')
    fig7.update_yaxes(categoryorder='array', categoryarray=['None', 'Instrument 1', 'Instrument 2', 'Both instruments'])

    norm_tot_rep_number = abs(sum(list(map(rep_count, ranges_dec))) - sum(list(map(rep_count, ranges_inc))))*72 / len(
        timeseries[surgery_number])
    max_rep_number = max(list(map(rep_count, ranges_dec)))
    autocorr_t1 = acf_list[surgery_number][1]
    tot_rep_number = abs(sum(list(map(rep_count, ranges_dec))) - sum(list(map(rep_count, ranges_inc))))

    indicators_1 = go.Figure()
    indicators_1.add_trace(go.Indicator(
        mode="number",
        value=norm_tot_rep_number,
        number={'suffix': ""},
        title={"text": "<br><span style='font-size:0.9em;color:gray'>Instrument Swaps/h</span>"},
        domain={'row': 0, 'column': 0}))

    indicators_1.add_trace(go.Indicator(
        mode="number",
        value=max_rep_number,
        number={'suffix': ""},
        title={"text": "<span style='font-size:0.9em;color:gray'>Max Consecutive <br> Repetitions</span>"},
        domain={'row': 1, 'column': 0}))

    indicators_1.update_layout(
        grid={'rows': 2, 'columns': 2, 'pattern': "independent"},
        margin=dict(l=50, r=50, t=30, b=30)
    )

    indicators_2 = go.Figure()
    indicators_2.add_trace(go.Indicator(
        mode="number",
        value=autocorr_t1,
        number={'suffix': ""},
        title={"text": "<br><span style='font-size:0.9em;color:gray'>Autocorrelation Lag 1</span>"},
        domain={'row': 1, 'column': 0}))

    indicators_2.add_trace(go.Indicator(
        mode="number",
        value=tot_rep_number,
        number={'suffix': ""},
        title={"text": "<span style='font-size:0.9em;color:gray'>Total Repetitions</span>"},
        domain={'row': 0, 'column': 0}))

    indicators_2.update_layout(
        grid={'rows': 2, 'columns': 2, 'pattern': "independent"},
        margin=dict(l=50, r=50, t=30, b=30)
    )

    account_name = accounts[surgery_number]

    return fig1, fig2, fig3, fig4, fig5, fig6, fig7, indicators_1, indicators_2


fig1, fig2, fig3, fig4, fig5, fig6, fig7, indicators_1, indicators_2 = autocorr_dash(
    seq='Medium Compression', surgery_number='Procedure 5')
surgery_dropdown = dcc.Dropdown(options=[f'Procedure {str(i+1)}' for i in range(len(timeseries_compressed_10k))],
                                id='surgery_number',
                                clearable=False,
                                value='Procedure 5', className="dbc",
                                placeholder='Select a Surgery', maxHeight=100)

smoothing_dropdown = dcc.Dropdown(options=['Low Compression', 'Medium Compression', 'High Compression'],
                                  id='smoothing_name',
                                  clearable=False, className="dbc",
                                  value='Medium Compression',
                                  placeholder='Select a Smoothness')

def Navbar():
    layout = html.Div([
        dbc.NavbarSimple(
            brand="Mining Surgical Instrument Annotations",
            id="navbar",
            color="dark",
            dark=True,
            brand_style={'fontSize': '30px', 'textAlign': 'center', 'width': '100%'}
        ),
    ])

    return layout

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
                meta_tags=[{"name": "viewport", "content": "width=device-width"}])
server = app.server

navbar_layout = Navbar()
app.layout = dbc.Container(
    [navbar_layout,
     # dbc.Row(dbc.Col(html.H2('Autocorrelation Clustering Analysis', className='text-center text-primary, mb-3'))),
     #dbc.Row(dbc.Col(html.H3(id='account_name', children=account_name, className='text-center text-primary, mb-3'))),
     dbc.Row([dbc.Col(surgery_dropdown),
              dbc.Col(smoothing_dropdown)]),
     dbc.Row([
         dbc.Col([
             dcc.Graph(id='fig1', figure=fig1,
                       style={'height': 500}),
             html.Hr()
         ], width={'size': 6, 'offset': 0, 'order': 1}),
         dbc.Col([
             dcc.Graph(id='fig6', figure=fig6,
                       style={'height': 500}),
             html.Hr()
         ], width={'size': 6, 'offset': 0, 'order': 2})],
         align="center"),

     dbc.Row([
         dbc.Col([
             dcc.Graph(id='fig2', figure=fig2,
                       style={'height': 300}),
             html.Hr()
         ], width={'size': 7, 'offset': 0, 'order': 1}),
         dbc.Col([
             dcc.Graph(id='fig5', figure=fig5,
                       style={'height': 400}),
             html.Hr()
         ], width={'size': 5, 'offset': 0, 'order': 2})]
         , align="center"),

     dbc.Row([
         dbc.Col([
             dcc.Graph(id='fig7', figure=fig7,
                       style={'height': 350}),
             html.Hr(),
         ], width={'size': 7, 'offset': 0, 'order': 1}),
         dbc.Col([
             dcc.Graph(id='indicators_1', figure=indicators_1,
                       style={'height': 350}),
             html.Hr(),
         ], width={'size': 2, 'offset': 0, 'order': 2}),
         dbc.Col([
             dcc.Graph(id='indicators_2', figure=indicators_2,
                       style={'height': 350}),
             html.Hr(),
         ], width={'size': 3, 'offset': 0, 'order': 3}),
     ], align="center"),

     dbc.Row([
         dbc.Col([
             dcc.Graph(id='fig3', figure=fig3,
                       style={'height': 300}),
             html.Hr()
         ], width={'size': 7, 'offset': 0, 'order': 1}),
         dbc.Col([
             dcc.Graph(id='fig4', figure=fig4,
                       style={'height': 300}),
             html.Hr()
         ], width={'size': 5, 'offset': 0, 'order': 2})
     ], align="center")
     ], fluid=True)


@app.callback([Output(component_id="fig1", component_property="figure"),
               Output(component_id="fig2", component_property="figure"),
               Output(component_id="fig3", component_property="figure"),
               Output(component_id="fig4", component_property="figure"),
               Output(component_id="fig5", component_property="figure"),
               Output(component_id="fig6", component_property="figure"),
               Output(component_id="fig7", component_property="figure"),
               Output(component_id="indicators_1", component_property="figure"),
               Output(component_id="indicators_2", component_property="figure")],
              [Input(component_id="smoothing_name", component_property="value"),
               Input(component_id="surgery_number", component_property="value")])
def callback_function(smoothing_name, surgery_number):
    fig1, fig2, fig3, fig4, fig5, fig6, fig7, indicators_1, indicators_2 = autocorr_dash(
        seq=smoothing_name, surgery_number=surgery_number)
    return fig1, fig2, fig3, fig4, fig5, fig6, fig7, indicators_1, indicators_2


if __name__ == "__main__":
    app.run_server(debug=False, port=8053)
