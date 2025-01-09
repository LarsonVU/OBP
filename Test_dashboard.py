import dash
from dash import dcc, html
import dash_bootstrap_components as dbc

# Create the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Sidebar layout
def sidebar():
    return html.Div([
        html.Div([
            html.Img(
                src="https://uniim1.shutterfly.com/ng/services/mediarender/THISLIFE/021036514417/media/23148907008/medium/1501685726/enhance",
                className="rounded-circle",
                style={"width": "65px"},
            ),
            html.Div([
                html.H5("Jone Doe", className="text-light"),
                html.P("Lorem ipsum dolor sit amet consectetur.", className="text-muted"),
            ], className="ms-2")
        ], className="d-flex align-items-center px-3 py-4"),

        html.Div([
            dbc.Input(placeholder="Search here", type="text", className="form-control w-100 bg-transparent"),
        ], className="px-4 py-3 mt-2"),

        dbc.Nav([
            dbc.NavLink([html.I(className="uil-estate"), " Dashboard"], href="#"),
            dbc.NavLink([html.I(className="uil-folder"), " File Manager"], href="#"),
            dbc.NavLink([html.I(className="uil-calendar-alt"), " Calendar"], href="#"),
            dbc.NavLink([html.I(className="uil-envelope-download"), " Mailbox"], href="#"),
            dbc.NavLink([html.I(className="uil-shopping-cart-alt"), " Ecommerce"], href="#"),
            dbc.NavLink([html.I(className="uil-bag"), " Projects"], href="#"),
            dbc.NavLink([html.I(className="uil-setting"), " Settings"], href="#"),
            dbc.NavLink([html.I(className="uil-chart-pie-alt"), " Components"], href="#"),
            dbc.NavLink([html.I(className="uil-panel-add"), " Charts"], href="#"),
            dbc.NavLink([html.I(className="uil-map-marker"), " Maps"], href="#"),
        ], vertical=True, pills=True, className="nav-pills flex-column"),
    ], className="sidebar bg-dark text-white p-3")

# Content layout
def content():
    return html.Div([
        html.Nav([
            html.Div([
                html.Button(html.I(className="uil-bars text-white"), className="navbar-toggler"),
                html.A("adminkit", className="navbar-brand text-light"),
            ], className="d-flex align-items-center"),

            dbc.Nav([
                dbc.DropdownMenu([
                    dbc.DropdownMenuItem("My Account", href="#"),
                    dbc.DropdownMenuItem("My Inbox", href="#"),
                    dbc.DropdownMenuItem("Help", href="#"),
                    dbc.DropdownMenuItem(divider=True),
                    dbc.DropdownMenuItem("Log Out", href="#"),
                ], label="Settings", className="text-light"),
                dbc.NavItem(dbc.NavLink(html.I(className="uil-comments-alt"), href="#")),
                dbc.NavItem(dbc.NavLink(html.I(className="uil-bell"), href="#")),
            ], className="ms-auto")
        ], className="navbar navbar-expand-md bg-dark"),

        html.Div([
            html.Div([
                html.H1("Welcome to Dashboard", className="text-light fs-3"),
                html.P("Hello Jone Doe, welcome to your awesome dashboard!", className="text-muted"),
            ], className="p-3 rounded bg-secondary"),

            html.Div([
                html.Div([
                    html.Div([
                        html.I(className="uil-envelope-shield bg-primary text-light fs-3"),
                        html.Div([
                            html.H3("1,245", className="mb-0"),
                            html.Span("Emails", className="text-muted ms-2"),
                        ], className="ms-3")
                    ], className="d-flex align-items-center p-3 rounded bg-dark"),
                ], className="col-lg-4 mb-4"),

                html.Div([
                    html.Div([
                        html.I(className="uil-file bg-danger text-light fs-3"),
                        html.Div([
                            html.H3("34", className="mb-0"),
                            html.Span("Projects", className="text-muted ms-2"),
                        ], className="ms-3")
                    ], className="d-flex align-items-center p-3 rounded bg-dark"),
                ], className="col-lg-4 mb-4"),

                html.Div([
                    html.Div([
                        html.I(className="uil-users-alt bg-success text-light fs-3"),
                        html.Div([
                            html.H3("5,245", className="mb-0"),
                            html.Span("Users", className="text-muted ms-2"),
                        ], className="ms-3")
                    ], className="d-flex align-items-center p-3 rounded bg-dark"),
                ], className="col-lg-4 mb-4"),
            ], className="row"),
        ], className="p-4")
    ])

# App layout
app.layout = html.Div([
    dbc.Row([
        dbc.Col(sidebar(), width=3),
        dbc.Col(content(), width=9)
    ])
], className="bg-dark")

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
