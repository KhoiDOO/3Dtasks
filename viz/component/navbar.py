import dash_bootstrap_components as dbc

def get_navbar(name:str):
    return dbc.NavbarSimple(
        brand=name,
        brand_href="#",
        color="dark",
        dark=True,
    )