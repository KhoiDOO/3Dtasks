import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import os

import dash
import dash_bootstrap_components as dbc
from dash import html
from dash import dcc
import argparse

from dash.dependencies import Input, Output, State

from component import get_navbar
from utils import read_off, build_mesh_graph, build_pcloud_graph, PointSampler, point_sampler

def list_files(directory):
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            files.append(os.path.relpath(os.path.join(root, filename), directory))
    return files

# Cache for storing vertices and faces
cache = {}

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--colorscale', type=str, default='Rainbow')
    args = parser.parse_args()

    external_stylesheets = [dbc.themes.BOOTSTRAP]
    app = dash.Dash(__name__, update_title=None, external_stylesheets=external_stylesheets)
    server = app.server

    mesh_card = dbc.Card(
        [
            dbc.CardHeader("Mesh Representation"),
            dbc.CardBody([dcc.Graph(id="mesh-graph")]),
        ],
    )

    pcloud_card = dbc.Card(
        [
            dbc.CardHeader("Point Cloud Representation"),
            dbc.CardBody([dcc.Graph(id="pcloud-graph")]),
        ],
    )

    app.layout = html.Div(
        [
            get_navbar(),
            dbc.Container(
                [
                    dbc.Row([
                        dbc.Col([
                            dcc.Input(id='directory-input', type='text', placeholder='Enter directory path', value='../data/src'),
                            html.Button('Load Directory', id='load-directory-button', n_clicks=0),
                            dcc.Dropdown(id='file-dropdown', placeholder="Select a file"),
                            html.Button('Load Model', id='load-model-button', n_clicks=0),
                            dcc.Slider(
                                id='point-slider',
                                min=100,
                                max=10000,
                                step=100,
                                value=1000,
                                marks={i: str(i) for i in range(100, 10001, 1000)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ])
                    ]),
                    dbc.Row([dbc.Col([mesh_card]), dbc.Col([pcloud_card])])
                ], fluid=True
            )
        ]
    )

    @app.callback(
        Output('file-dropdown', 'options'),
        Input('load-directory-button', 'n_clicks'),
        State('directory-input', 'value')
    )
    def update_file_dropdown(n_clicks, directory):
        if directory and os.path.isdir(directory):
            files = list_files(directory)
            return [{'label': f, 'value': f} for f in files]
        return []

    @app.callback(
        [Output('mesh-graph', 'figure'), Output('pcloud-graph', 'figure')],
        [Input('load-model-button', 'n_clicks'), Input('point-slider', 'value')],
        [State('directory-input', 'value'), State('file-dropdown', 'value')]
    )
    def update_output(load_clicks, num_points, directory, filename):
        ctx = dash.callback_context

        if not ctx.triggered:
            return {}, {}

        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if directory and filename:
            filepath = os.path.join(directory, filename)
            if filepath not in cache:
                vertices, faces = read_off(filepath)
                cache[filepath] = (vertices, faces)
            else:
                vertices, faces = cache[filepath]

            if trigger_id == 'load-model-button':
                mesh_fig = build_mesh_graph(vertices, faces, colorscale=args.colorscale)
                sample_points = point_sampler(vertices, faces, num_points)
                pcloud_fig = build_pcloud_graph(sample_points, colorscale=args.colorscale)
                return mesh_fig, pcloud_fig

            elif trigger_id == 'point-slider':
                sample_points = point_sampler(vertices, faces, num_points)
                pcloud_fig = build_pcloud_graph(sample_points, colorscale=args.colorscale)
                return dash.no_update, pcloud_fig

        return {}, {}

    app.run(debug=True, dev_tools_props_check=False)