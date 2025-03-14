from submodules.mvdream_diffusers.pipeline_mvdream import MVDreamPipeline

import dash
import dash_bootstrap_components as dbc
import torch
import numpy as np

from dash import html
from dash import dcc
from dash.dependencies import Input, Output, State

from viz.component import get_navbar
from viz.utils import array_to_base64

# Initialize the app
external_stylesheets = [dbc.themes.BOOTSTRAP]
app = dash.Dash(__name__, update_title=None, external_stylesheets=external_stylesheets)
server = app.server

# Layout
app.layout = html.Div(
    [
        get_navbar(name="MVDream Viewer"),
        dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dcc.Input(
                                    id="text-prompt",
                                    type="text",
                                    placeholder="Enter text prompt",
                                    style={"width": "100%"},
                                ),
                                dcc.Dropdown(
                                    id="model-dropdown",
                                    options=[
                                        {"label": "ashawkey/mvdream-sd2.1-diffusers", "value": "ashawkey/mvdream-sd2.1-diffusers"},
                                        {"label": "ashawkey/mvdream-sd2.0-diffusers", "value": "ashawkey/mvdream-sd2.0-diffusers"},
                                    ],
                                    placeholder="Select a model",
                                ),
                                dcc.Input(
                                    id="num-frames",
                                    type="number",
                                    placeholder="Enter number of frames",
                                    min=1,
                                    max=10,
                                    step=1,
                                    value=4,
                                ),
                                html.Button("Generate Frames", id="generate-button", n_clicks=0),
                            ],
                            width=4,
                        ),
                        dbc.Col(
                            [
                                html.Div(id="output-grid"),
                            ],
                            width=8,
                        ),
                    ]
                )
            ],
            fluid=True,
        ),
    ]
)

# Callback to generate frames
@app.callback(
    Output("output-grid", "children"),
    Input("generate-button", "n_clicks"),
    State("text-prompt", "value"),
    State("model-dropdown", "value"),
    State("num-frames", "value"),
)
def generate_frames(n_clicks, prompt, model, num_frames):
    if n_clicks > 0 and prompt and model and num_frames:
        # Load the pipeline
        pipe = MVDreamPipeline.from_pretrained(
            model,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        pipe = pipe.to("cuda")

        # Generate frames using the num_frames argument
        images = pipe(prompt, guidance_scale=5, num_inference_steps=30, num_frames=num_frames)

        # Create individual image views for each frame
        image_views = []
        for idx, image in enumerate(images):
            image_html = html.Div(
                [
                    html.Img(src=f"data:image/png;base64,{array_to_base64(image)}", style={"width": "100%"}),
                    html.P(f"Frame {idx + 1}", style={"text-align": "center"}),
                ],
                style={"margin-bottom": "20px"},
            )
            image_views.append(image_html)

        return dbc.Row([dbc.Col(image_view, width=6) for image_view in image_views])

    return html.Div("Enter all inputs and click 'Generate Frames'.")


if __name__ == "__main__":
    app.run_server(debug=True)