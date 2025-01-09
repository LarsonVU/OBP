from dash import Dash, dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import base64
import io
import plotly.express as px


# Initialize the app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
app.title = "Excel Upload Dashboard"

dark_layout = {
    'plot_bgcolor': '#1e1e1e',  # Dark background for the plot
    'paper_bgcolor': '#1e1e1e',  # Dark background for the whole page
    'font': {'color': 'white'},   # White text
    'xaxis': {'showgrid': False},  # Hide the grid
    'yaxis': {'showgrid': False},  # Hide the grid
    'colorway': ['#636EFA', '#EF553B', '#00CC96', '#AB63A1'],  # Default color palette
}

sidebar = html.Div(
    [
        html.Img(
            src = "assets/MS solutions wit.png",  # Replace "logo.png" with your image name
            alt="Logo",
            style={
                "width": "150px",  # Adjust width of the logo as needed
                "margin": "0 auto",  # Center the logo horizontally
                "display": "block",  # Ensure the image is treated as a block element
            },
        ),
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
    id="sidebar",
    style={
        "backgroundColor": "#343a40",  # Dark sidebar background
        "padding": "20px",
        "height": "100vh",  # Full-height sidebar
        "width": "250px",  # Sidebar width
        "position": "fixed",
        "transition": "width 0.3s ease",  # Smooth transition for the width change
        "display": "flex",
        "flexDirection": "column",
        "alignItems": "center",  # Align items in the center
        "justifyContent": "start",  # Align content at the start
    },
)


# Sidebar Toggle Button
toggle_button = html.Button(
    "<",  # Initially show "+"
    id="toggle-btn",
    style={
        "position": "absolute",
        "top": "50%",
        "left": "251px",  # Position it to the right of the sidebar
        "backgroundColor": "transparent",
        "color": "black",
        "border": "none",
        "padding": "10px",
        "cursor": "pointer",
        "fontSize": "20px",
        "zIndex": "1000",
    }
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
    [
        dcc.Store(id="sidebar-state", data="open"),  # Initialize the sidebar state to "open"
        dcc.Location(id="url"),
        sidebar,
        toggle_button,
        content
    ]
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

# File Input Section Layout (Updated with provided code)
def file_input_layout():
    return html.Div([
        html.H1("Excel File Uploader", style={"textAlign": "center"}),

        dcc.Upload(
            id='upload-data',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select an Excel File', style={'color': 'blue'})
            ]),
            style={
                'width': '80%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '2px',
                'borderStyle': 'solid',
                'borderRadius': '10px',
                'textAlign': 'center',
                'margin': 'auto'
            },
            multiple=False
        ),

        html.Div(id='output-data-upload'),

        html.Button(
            "Add Row",
            id="add-row-btn",
            n_clicks=0,
            style={"display": "block", "margin": "20px auto"}
        ),
    ])

# Callback to parse and display the uploaded file
@app.callback(
    Output('output-data-upload', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'))
def display_table(contents, filename):
    if contents is None:
        return html.Div("The presented jobs will be displayed here", style={'textAlign': 'center', 'font-family' : 'Roboto' })

    content_type, content_string = contents.split(',')

    try:
        # Decode the uploaded file
        decoded = base64.b64decode(content_string)
        if filename.endswith('.xlsx'):
            # Read the Excel file
            df = pd.read_excel(io.BytesIO(decoded))

            if len(df.columns) > 5:
                process_time_titles = [f"Process time {i+1}" for i in range(len(df.columns )-4)]
            else:
                return html.Div("Unsupported file type. Please upload an Excel file with service times dedicated to each machine")
            titles = ['Job ID', 'Release Date', 'Due Date', 'Weight'] +process_time_titles
            df = df.set_axis(titles, axis=1)

            # Store the data in a global variable for graph updates
            app.layout.df = df  # Store the dataframe in the app layout

            # Return the data table
            return html.Div([
                html.H3(f"Uploaded File: {filename}", style={'textAlign': 'center', 'font-family' : 'Roboto'}),
                dash_table.DataTable(
                    id="sortable-table",
                    columns=[{"name": col, "id": col, "deletable": False} for col in df.columns],
                    data=df.to_dict("records"),
                    sort_action="native",  # Enables sorting by clicking column headers
                    style_table={"overflowX": "auto"},
                    style_header={
                        "backgroundColor": "rgb(130, 130, 230)",
                        "fontWeight": "bold",
                    },
                    style_data={
                        "backgroundColor": "black",  # Light gray background for rows
                        "color": "white",
                    },
                    style_cell={"textAlign": "center", "padding": "5px"},
                    editable=True,
                    row_deletable=True
                ),
            ])
        else:
            return html.Div("Unsupported file type. Please upload an Excel file.")
    except Exception as e:
        return html.Div(f"There was an error processing the file: {str(e)}")

@app.callback(Output('sortable-table', 'data', allow_duplicate=True),
    Input('add-row-btn', 'n_clicks'),
    State('sortable-table', 'data'),
    State('sortable-table', 'columns'),
    prevent_initial_call=True
)
def add_row(n_clicks, rows, columns):
    if rows is None:
        rows = []
    new_row = {col['id']: '' for col in columns}  # Create an empty row
    rows.append(new_row)  # Append the new row
    return rows

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
    # Ensure that df is available (if the file has been uploaded)
    df = getattr(app.layout, 'df', None)
    if df is not None:
        # Create the graphs based on the uploaded data
        graph1 = dcc.Graph(
            figure=px.scatter(
                df, x="Weight", y="Process time 1", color="Job ID", title="Weight vs Process Time 1"
            ).update_layout(dark_layout),
            config={"displayModeBar": False},
        )

        graph2 = dcc.Graph(
            figure=px.scatter(
                df, x="Weight", y="Process time 2", color="Job ID", title="Weight vs Process Time 2"
            ).update_layout(dark_layout),
            config={"displayModeBar": False},
        )

        graph3 = dcc.Graph(
            figure=px.scatter(
                df, x="Process time 1", y="Process time 2", color="Job ID", title="Process Time 1 vs Process Time 2"
            ).update_layout(dark_layout),
            config={"displayModeBar": False},
        )

        graph4 = dcc.Graph(
            figure=px.scatter(
                df, x="Process time 2", y="Process time 3", color="Job ID", title="Process Time 2 vs Process Time 3"
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
    else:
        return html.Div("No data uploaded. Please upload a file first.")


@app.callback(
    [Output("sidebar", "style"),
     Output("page-content", "style"),
     Output("toggle-btn", "children"),
     Output("toggle-btn", "style"),  # Add an output to modify the button style
     Output("sidebar-state", "data", allow_duplicate=True)],
    Input("toggle-btn", "n_clicks"),
    State("sidebar-state", "data"),
    prevent_initial_call='initial_duplicate'  # Add this line to allow duplicate callbacks
)
def toggle_sidebar(n_clicks, current_state):
    if n_clicks is None:
        # Return default values when the button has not been clicked yet
        return (
            {
                "backgroundColor": "#343a40",
                "padding": "20px",
                "height": "100vh",
                "width": "250px",  # Sidebar visible
                "position": "fixed",
                "overflow": "auto",
                "display": "flex",
                "flexDirection": "column",  # Stack items vertically
            },
            {"marginLeft": "270px", "padding": "20px", "backgroundColor": "#1e1e1e", "minHeight": "100vh", "color": "white"},
            "<",  # Default button is "<"
            {"position": "absolute", "top": "50%", "left": "242px", "backgroundColor": "transparent", "color": "black", "border": "none", "padding": "10px", "cursor": "pointer", "fontSize": "20px", "zIndex": "1000"},
            "open"  # Default state is "open"
        )
    
    if current_state == "open":
        # Close the sidebar (make it disappear, but keep a small width for the button)
        return (
            {"backgroundColor": "#343a40", "padding-top": "20px", "height": "100vh", "width": "0px", "position": "fixed", "overflow": "hidden"},
            {"marginLeft": "20px", "padding-top": "20px", "backgroundColor": "#1e1e1e", "minHeight": "100vh", "color": "white"},
            ">",  # Change button to ">"
            {"position": "absolute", "top": "50%", "left": "5px", "backgroundColor": "transparent", "color": "black", "border": "none", "padding": "0px", "cursor": "pointer", "fontSize": "20px", "zIndex": "1000"},
            "closed"  # Update sidebar state to closed
        )
    else:
        # Open the sidebar (make it visible)
        return (
            {
                "backgroundColor": "#343a40",
                "padding": "20px",
                "height": "100vh",
                "width": "250px",  # Sidebar visible
                "position": "fixed",
                "overflow": "auto",
                "display": "flex",
                "flexDirection": "column",  # Stack items vertically
            },
            {"marginLeft": "270px", "padding": "20px", "backgroundColor": "#1e1e1e", "minHeight": "100vh", "color": "white"},
            "<",  # Default button is "<"
            {"position": "absolute", "top": "50%", "left": "242px", "backgroundColor": "transparent", "color": "black", "border": "none", "padding": "10px", "cursor": "pointer", "fontSize": "20px", "zIndex": "1000"},
            "open"  # Default state is "open"
        )


# Run the app
if __name__ == "__main__":
    app.run_server(debug=False)
