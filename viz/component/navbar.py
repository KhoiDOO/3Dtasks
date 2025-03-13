import dash_bootstrap_components as dbc
from dash import html

nav_bar = dbc.NavbarSimple(
    brand="3D Viewer",
    brand_href="#",
    color="dark",
    dark=True,
)

def get_navbar():
    return nav_bar