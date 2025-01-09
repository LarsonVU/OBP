from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px

# Initialize the Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Sample data for the graphs
df = px.data.iris()

# Define common layout for dark mode
dark_layout = {
    "plot_bgcolor": "#2c2c2c",
    "paper_bgcolor": "#2c2c2c",
    "font": {"color": "white"},
}

# Sidebar layout
sidebar = html.Div(
    [
        html.H2("Navigation", className="text-light text-center"),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink("File Input", href="/file-input", active="exact"),
                dbc.NavLink("Algorithm Settings", href="/algorithm-settings", active="exact"),
                dbc.NavLink("Graphs Section", href="/graphs", active="exact"),
            ],
            vertical=True,
            pills=True,
            className="text-light",
        ),
    ],
    style={
        "backgroundColor": "#343a40",  # Dark sidebar background
        "padding": "20px",
        "height": "100vh",  # Full-height sidebar
        "width": "250px",
        "position": "fixed",
    },
)

# Content layout
content = html.Div(
    id="page-content",
    style={
        "marginLeft": "270px",  # Space for the sidebar
        "padding": "20px",
        "backgroundColor": "#1e1e1e",
        "minHeight": "100vh",
        "color": "white",
    },
)

# App layout with sidebar and content
app.layout = html.Div(
    [dcc.Location(id="url"), sidebar, content]
)

# Callbacks to handle page navigation
@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname"),
)
def display_page(pathname):
    if pathname == "/file-input":
        return file_input_layout()
    elif pathname == "/algorithm-settings":
        return algorithm_settings_layout()
    elif pathname == "/graphs":
        return graphs_layout()
    else:
        return html.H1("Welcome to the App", className="text-center")

# File Input Section Layout
def file_input_layout():
    return html.Div(
        [
            html.H3("File Input", className="text-center"),
            dcc.Upload(
                id="file-upload",
                children=html.Div(["Drag and Drop or ", html.A("Select a File")]),
                style={
                    "width": "100%",
                    "height": "60px",
                    "lineHeight": "60px",
                    "borderWidth": "1px",
                    "borderStyle": "dashed",
                    "borderRadius": "5px",
                    "textAlign": "center",
                    "margin": "10px",
                },
            ),
            html.Div(id="file-output", className="mt-3"),
        ]
    )

# Algorithm Settings Section Layout
def algorithm_settings_layout():
    return html.Div(
        [
            html.H3("Algorithm Settings", className="text-center"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label("Parameter 1:", className="mt-2"),
                            dcc.Input(
                                id="param-1",
                                type="number",
                                placeholder="Enter value",
                                className="form-control",
                            ),
                        ],
                        width=6,
                    ),
                    dbc.Col(
                        [
                            html.Label("Parameter 2:", className="mt-2"),
                            dcc.Input(
                                id="param-2",
                                type="number",
                                placeholder="Enter value",
                                className="form-control",
                            ),
                        ],
                        width=6,
                    ),
                ]
            ),
            dbc.Button("Run Algorithm", id="run-btn", color="primary", className="mt-3"),
            html.Div(id="algorithm-output", className="mt-3"),
        ]
    )

# Graphs Section Layout
def graphs_layout():
    graph1 = dcc.Graph(
        figure=px.scatter(
            df, x="sepal_width", y="sepal_length", color="species", title="Scatter Plot"
        ).update_layout(dark_layout),
        config={"displayModeBar": False},
    )

    graph2 = dcc.Graph(
        figure=px.bar(
            df, x="species", y="sepal_width", title="Bar Chart"
        ).update_layout(dark_layout),
        config={"displayModeBar": False},
    )

    graph3 = dcc.Graph(
        figure=px.line(
            df, x="sepal_length", y="petal_length", title="Line Chart"
        ).update_layout(dark_layout),
        config={"displayModeBar": False},
    )

    graph4 = dcc.Graph(
        figure=px.histogram(
            df, x="petal_width", title="Histogram"
        ).update_layout(dark_layout),
        config={"displayModeBar": False},
    )

    return html.Div(
        [
            html.H3("Graphs Section", className="text-center"),
            dbc.Row(
                [
                    dbc.Col(graph1, width=6),
                    dbc.Col(graph2, width=6),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(graph3, width=6),
                    dbc.Col(graph4, width=6),
                ]
            ),
        ]
    )

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
