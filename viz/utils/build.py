
import plotly.graph_objects as go
import numpy as np

def build_mesh_graph(vertices, faces, opacity=0.2, colorscale ='Rainbow'):
    x, y, z = vertices
    i, j, k = faces

    intensity = np.linspace(0, 1, len(i))

    fig_mesh = go.Figure(layout=go.Layout(height=800, width=800))
    fig_mesh.add_trace(
        go.Mesh3d(
            x=x, y=y, z=z, 
            opacity=opacity, 
            i=i, j=j, k=k,
            intensity=intensity,
            colorscale=colorscale,
        )
    )

    return fig_mesh

def build_pcloud_graph(vertices, opacity=0.2, colorscale ='Rainbow'):
    x, y, z = vertices

    fig_pcloud = go.Figure(layout=go.Layout(height=800, width=800))
    fig_pcloud.add_trace(
        go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(
                size=2,
                opacity=opacity,
                color=x,
                colorscale=colorscale
            )
        )
    )

    return fig_pcloud