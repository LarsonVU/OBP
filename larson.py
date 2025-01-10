from dash import Dash, dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import base64
import io
import plotly.express as px
import ilp_algorithm as ilp
import plotly.graph_objects as go
from plotly.colors import n_colors



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

header = html.Div(
            [
                html.Div(
                    "☰",  # Hamburger icon
                    id="hamburger-menu",
                    style={
                        "fontSize": "24px",
                        "cursor": "pointer",
                        "marginRight": "20px",
                        "color": "white",
                    },
                ),
                html.Img(
                    src="assets/MS solutions wit.png",  # Replace with your logo path
                    alt="Logo",
                    style={
                        "height": "40px",
                    },
                ),
            ],
            style={
                "display": "flex",
                "alignItems": "center",
                "backgroundColor": "#1C4E80",
                "padding": "10px 20px",
                "color": "white",
                "position": "fixed",
                "width": "100%",
                "zIndex": 1,
            },
        )

sidebar = html.Div(
    [
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
            "backgroundColor": "#1C4E80",
            "padding-top": "20px", 
            "height": "100vh", 
            "width": "0px", 
            "position": "fixed", 
            "overflow": "hidden",
    },
)

# Content layout
content = html.Div(
    id="page-content",
    style={
        "marginLeft": "0px",  # Space for the sidebar
        "padding": "100px 20px 0px 20px",
        "backgroundColor": "#F1F1F1",
        "fontFamily": "montserrat, sans-serif",
        "minHeight": "100vh",
        "color": "white",
    },
)


# App layout with sidebar and content
app.layout = html.Div(
    [
        dcc.Store(id="sidebar-state", data="close"),  # Initialize the sidebar state to "open"
        dcc.Location(id="url"),
        header,
        sidebar,
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
        return file_input_layout()

# File Input Section Layout (Updated with provided code)
def file_input_layout():
    return html.Div(
        [
            html.H1("Upload your excel file", style={"textAlign": "center"}),

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
                style={
                    "display": "none",  # Initially hidden
                    "margin": "20px auto",
                    "backgroundColor": "#1C4E80",  # Button background color
                    "color": "#F1F1F1",  # Button text color
                    "border": "2px solid #7E909A",  # Border color
                    "padding": "10px 20px",  # Padding inside the button
                    "borderRadius": "8px",  # Rounded corners
                    "fontSize": "16px",  # Font size
                    "fontWeight": "bold",  # Bold text
                    "cursor": "pointer",  # Pointer cursor on hover
                    "transition": "background-color 0.3s ease",  # Smooth hover transition
                }
            )
        ],
        style={  # Moved the style attribute here
            'backgroundColor': '#FFF',
            'padding': '30px 20px',
            'color': 'black',
            'display': 'flex',
            'flexDirection': 'column',
            'justifyContent': 'space-between',
            'gap': '20px',
        },
    )


# Callback to parse and display the uploaded file
from dash import dcc, html, Input, Output, State, callback

# Callback to parse and display the uploaded file
@app.callback(
    [
        Output('output-data-upload', 'children'),
        Output('add-row-btn', 'style')  # Add output to modify button style
    ],
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def display_table(contents, filename):
    if contents is None:
        # If no file is uploaded, keep button hidden and display placeholder text
        return html.Div(
            "The presented jobs will be displayed here", 
            style={'textAlign': 'center', 'font-family': 'Roboto'}
        ), {"display": "none"}

    content_type, content_string = contents.split(',')

    try:
        # Decode the uploaded file
        decoded = base64.b64decode(content_string)
        if filename.endswith('.xlsx'):
            # Read the Excel file
            df = pd.read_excel(io.BytesIO(decoded))

            if len(df.columns) > 5:
                process_time_titles = [f"Process time {i+1}" for i in range(len(df.columns) - 4)]
            else:
                return html.Div(
                    "Unsupported file type. Please upload an Excel file with service times dedicated to each machine"
                ), {"display": "none"}
            
            titles = ['Job ID', 'Release Date', 'Due Date', 'Weight'] + process_time_titles
            df = df.set_axis(titles, axis=1)

            # Store the data in a global variable for graph updates
            app.layout.df = df  # Store the dataframe in the app layout

            # Return the data table and make the button visible
            return html.Div([
                html.H3(f"Uploaded File: {filename}", style={'textAlign': 'center', 'font-family': 'Roboto'}),
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
            ]),  {   "display": "block",  # Initially hidden
                    "margin": "20px auto",
                    "backgroundColor": "transparent",  # Button background color
                    "color": "#1C4E80",  # Button text color
                    "border": "1px solid #7E909A",  # Border color
                    "padding": "5px 20px",  # Padding inside the button
                    "borderRadius": "0px",  # Rounded corners
                    "fontSize": "16px",  # Font size
                    "fontWeight": "bold",  # Bold text
                    "fontFamily": "montserrat, sans-serif",  # Font family
                    "cursor": "pointer",  # Pointer cursor on hover
                    "transition": "background-color 0.3s ease", } # Smooth hover transition  # Make the button visible
        else:
            return html.Div("Unsupported file type. Please upload an Excel file."), {"display": "none"}
    except Exception as e:
        return html.Div(f"There was an error processing the file: {str(e)}"), {"display": "none"}



@app.callback(
    Output('placeholder', 'children'),
    Input('sortable-table', 'data'),
    State('sortable-table', 'columns'),
    prevent_initial_call = True)
def update_input_data(rows, columns):
    app.layout.df = pd.DataFrame(rows, columns=[col['name'] for col in columns]).astype(int)
    return 'hoi'

@app.callback(Output('sortable-table', 'data', allow_duplicate=True),
    Input('add-row-btn', 'n_clicks'),
    State('sortable-table', 'data'),
    State('sortable-table', 'columns'),
    prevent_initial_call=True
)
def add_row(n_clicks, rows, columns):
    if rows is None:
        rows = []
    new_row = {col['id']: 0 for col in columns}  # Create an empty row
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
                            html.Label("Max Runtime:", className="mt-2"),
                            dcc.Input(
                                id="max-runtime",
                                type="number",
                                placeholder="Enter value",
                                className="form-control",
                            ),
                        ],
                        width=6,
                    ),
                    dbc.Col(
                        [
                            html.Label("Algorithm-status", className="mt-2"),
                            html.Div(f"Not running yet", id = 'param-2'),
                        ],
                        width=6,
                    ),
                ]
            ),
            dbc.Button("Run Algorithm", id='run-btn', color="primary", className="mt-3", n_clicks= 0),
            html.Div([dbc.Button("Download Solution", id="btn-algorithm-output", className="mt-3", style={"display": "none"}), dcc.Download(id="schedule-excel")]) ]
    )

@app.callback(Output('param-2', 'children'),
              Input('run-btn', 'n_clicks'),
              prevent_initial_call=True)
def is_algorithm_running(n_clicks):
    return f"Running"

@app.callback(  
        Output('param-2', 'children', allow_duplicate=True),
        Output('btn-algorithm-output', 'style'),
        Input('run-btn', 'n_clicks'),
        State('max-runtime', 'value'),
        prevent_initial_call=True
)
def run_specified_algorithm(n_clicks, max_runtime):
    data = app.layout.df.copy()
    columns = ['job_id', 'release_date', 'due_date', 'weight'] + [f"st_{i+1}" for i in range(len(data.columns)-4)]
    data.columns = columns
    schedule, score, runtime = ilp.runAlgorithm(data, max_runtime)

    app.layout.schedule_df = schedule
    app.layout.schedule_stats = (score, runtime)

    return f"Finished", {"display": "block"}

@app.callback(
        Output('schedule-excel', 'data'),
        Input('btn-algorithm-output', 'n_clicks'),
        prevent_initial_call = True
)
def download_schedule(n_clicks):
    return dcc.send_data_frame(app.layout.schedule_df.to_excel, "schedule.xlsx", sheet_name="schedule")

def create_gannt_chart(schedule_df):
    fig = go.Figure()

    machines = int((len(schedule_df.columns)-1)/2)
    color_scheme = n_colors('rgb(234, 106, 71)', 'rgb(0, 145, 213)', len(schedule_df), colortype='rgb')
    app.layout.df = app.layout.df.sort_values(by = 'Release Date')
    job_ids = app.layout.df['Job ID'].tolist()
    job_colors = {job_id: color for job_id, color in zip(job_ids, color_scheme)}

    for index, row in schedule_df.iterrows():
        for m in range(machines):
            fig.add_trace(go.Bar(
                x=[row[f'Completion time machine {m+1}'] - row[f'Start time machine {m+1}']],
                y=[m+1],
                base=row[f'Start time machine {m+1}'],
                orientation='h',
                name=row['Job ID'],
                marker_color = job_colors[row['Job ID']],
                showlegend= False
            ))

    for job_id, color in job_colors.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, color=color),
            name=job_id
        ))

    fig.update_layout(
        title="Job Schedule for Machines",
        xaxis_title="Time",
        yaxis_title="Machine",
        barmode='stack',
        xaxis=dict(
            fixedrange=True  # Disable zooming
        ),
        yaxis=dict(
            tickmode='linear',
            tick0=1,
            dtick=1,
            fixedrange=True  # Disable panning/zooming
        ),
        showlegend=True,
    )
    return fig


# Graphs Section Layout
def graphs_layout():

    schedule_data = getattr(app.layout, 'schedule_df', None)
    if schedule_data is not None:
        schedule = dash_table.DataTable(
                id="schedule-table",
                columns=[{"name": col, "id": col, "deletable": False} for col in schedule_data.columns],
                data=app.layout.schedule_df.to_dict("records"),
                #sort_action="native",  # Enables sorting by clicking column headers
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
            )
        
        score, runtime = app.layout.schedule_stats

        display_runtime = html.Div([dbc.Row(html.Label("Runtime of the algorithm", className="mt-2")),
                                    dbc.Row(f"{runtime}")])
        display_score = html.Div([dbc.Row(html.Label("Score of the algorithm", className="mt-2")),
                                    dbc.Row(f"{score}")])
        schedule_graph = dcc.Graph(
            figure= create_gannt_chart(schedule_data), config={'staticPlot': True}
        )

        return html.Div(
            [
                html.H3("Graphs Section", className="text-center"),
                dbc.Row(
                    [
                        dbc.Col(schedule, width=12),
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(display_runtime, width=6),
                        dbc.Col(display_score, width= 6)
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(schedule_graph, width=12),
                    ]
                )
            ]
        )
    else:
        return html.H3("The schedule will be displayed here")



@app.callback(
    [Output("sidebar", "style"),
     Output("page-content", "style"),
     Output("hamburger-menu", "children"),  # Change icon of the hamburger menu
     Output("hamburger-menu", "style"),  # Modify the style of the hamburger menu
     Output("sidebar-state", "data")],  # Store the sidebar's state (open/closed)
    Input("hamburger-menu", "n_clicks"),  # Triggered by the hamburger menu
    State("sidebar-state", "data"),
    prevent_initial_call=True  # Avoid triggering the callback initially
)
def toggle_sidebar(n_clicks, current_state):
    if current_state == "open":
        # Close the sidebar
        return (
            {"backgroundColor": "#1C4E80", "padding-top": "20px", "height": "100vh", "width": "0px", "position": "fixed", "overflow": "hidden"},
            {"marginLeft": "0px", "padding-top": "100px", "backgroundColor": "#F1F1F1", "minHeight": "100vh", "color": "white"},
            "☰",  # Hamburger menu icon when sidebar is closed
            {"cursor": "pointer", "fontSize": "24px", "color": "white", "marginRight": "20px"},  # Hamburger menu style
            "closed"  # Update state to closed
        )
    else:
        # Open the sidebar
        return (
            {
                "backgroundColor": "#1C4E80",
                "padding": "40px 0px",
                "height": "100vh",
                "width": "200px",  # Sidebar visible
                "position": "fixed",
                "overflow": "auto",
                "display": "flex",
                "flexDirection": "column",  # Stack items vertically
            },
            {"marginLeft": "200px", "padding": "100px 10px", "backgroundColor": "#F1F1F1", "minHeight": "100vh", "color": "white"},
            "×",  # Close icon when the sidebar is open
            {"cursor": "pointer", "fontSize": "26px", "lineheight": "1.2", "color": "white", "marginRight": "30px"},  # Hamburger menu style
            "open"  # Update state to open
        )


# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
