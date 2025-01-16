from dash import Dash, dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import base64
import io
import plotly.express as px
import ilp_overtake as ilp
import plotly.graph_objects as go
import genetic as gen
from plotly.colors import n_colors
import dash



# TO DO Lijst tool:
# File
# Lege tabel laden aan het begin
# Job ID moet uniek zijn en laden bij add row (en als je job id 1,2,4 moet werken read input dingetje (lex zijn probleem))

# Algorithm settings
# Algorithm status needs to be expanded:
# 	Needs to look nicer
# 	Needs to add current score
# 	Needs to add remaining time
# 	Maybe a wheel?
# When returning to the algorithm settings after running the page should keep loading
# No buttons should be able to be pressed when algorithm is running

# Graphs Section
# Graph depicting path of score over time 
# the percentage gap should be displayed or it should say optimal


# Algemeen, code iets cleaner (maar ja boeie)


# Initialize the app
app = Dash(__name__, external_stylesheets=[
           dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
app.title = "MS Tools"



## Define styles for tables and buttons
table_layout = {
    "style_table": {
        "width": "100%",
        "overflowX": "auto",
        "margin": "20px 0",
        "borderRadius": "10px",
    },
    "style_header": {
        "backgroundColor": "var(--knoppen-blauw)",  # Dark blue
        "color": "var(--background-color)",  # Light text
        "fontWeight": "bold",
        "textAlign": "left",
        "border": "1px solid #7E909A",  # Orange border
    },
    "style_cell": {
        "backgroundColor": "var(--background-color)",  # Light background
        "color": "#202020",  # Dark text
        "textAlign": "left",
        "padding": "10px",
        "border": "1px solid #7E909A",  # Light border
    },
    "style_data": {
        "border": "1px solid #7E909A",  # Border for data rows
    },
    "style_data_conditional": [
        {
            "if": {"row_index": "odd"},
            "backgroundColor": "#A5D8DD",  # Subtle blue-green for alternate rows
        },
        {
            "if": {"state": "active"},  # Hover effect
            "backgroundColor": "#0091D5",  # Bright blue highlight
            "color": "var(--background-color)",  # Light text
        },
    ],
}


button_style1 = {"style": {
    "margin": "20px 0",
    "backgroundColor": "var(--knoppen-blauw)",  # Button background color
    "color": "#FFFFFF",  # Button text color
    "border": "none",  # No border
    "padding": "10px 20px",  # Padding inside the button
    "borderRadius": "5px",  # Rounded corners
    "fontSize": "16px",  # Font size
    "fontWeight": "bold",  # Bold text
    "fontFamily": "montserrat, sans-serif",  # Font family
    "cursor": "pointer",  # Pointer cursor on hover
    "transition": "background-color 0.3s ease",  # Smooth hover transition
}}

button_style2 = {"style" : {"display": "block",  # Initially hidden
                  "margin": "20px auto",
                  "backgroundColor": "transparent",  # Button background color
                  "color": "var(--knoppen-blauw)",  # Button text color
                  "border": "1px solid #7E909A",  # Border color
                  "padding": "5px 20px",  # Padding inside the button
                  "borderRadius": "0px",  # Rounded corners
                  "fontSize": "16px",  # Font size
                  "fontWeight": "bold",  # Bold text
                  "fontFamily": "montserrat, sans-serif",  # Font family
                  "cursor": "pointer",  # Pointer cursor on hover
                  "transition": "background-color 0.3s ease", }}  # Smooth hover transition  # Make the button visible}


unselected_button_style1 = {
    "style": {
        "margin": "20px 0",
        "backgroundColor": "#f0f0f0",  # Light background for unselected state
        "color": "#333333",  # Darker text for visibility
        "border": "1px solid #dcdcdc",  # Light border
        "padding": "10px 20px",
        "borderRadius": "5px",
        "fontSize": "16px",
        "fontWeight": "normal",  # Normal weight for unselected
        "fontFamily": "montserrat, sans-serif",
        "cursor": "pointer",
        "transition": "background-color 0.3s ease, color 0.3s ease",  # Smooth transitions
    }} 



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
        "backgroundColor": "var(--header-color)",
        "padding": "10px 20px",
        "color": "white",
        "position": "fixed",
        "width": "100%",
        "zIndex": 1,
        "borderBottom": "1px solid #fff",
    },
)

sidebar = html.Div(
    [
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink("File Input", href="/file-input", active="exact", style={
                            "color": "#FFF"}, n_clicks=0, id="file-input"),  # Change the color here
                dbc.NavLink("Algorithm Settings", href="/algorithm-settings",
                            active="exact", style={"color": "#FFF"}),
                dbc.NavLink("Graphs Section", href="/graphs",
                            active="exact", style={"color": "#FFF"}),
            ],
            vertical=True,
            pills=True,
            className="text-light",
        ),
    ],
    id="sidebar",
    style={
        "backgroundColor": "var(--header-color)",
        "padding-top": "20px",
        "height": "100vh",
        "width": "0px",
        "position": "fixed",
        "overflow": "hidden",
        "color": "white",
    },
)

# Content layout
content = html.Div(
    id="page-content",
    style={
        "marginLeft": "0px",  # Space for the sidebar
        "padding": "100px 20px 0px 20px",
        "backgroundColor": "var(--background-color)",
        "fontFamily": "montserrat, sans-serif",
        "minHeight": "100vh",
        "color": "black",
    },
)


# App layout with sidebar and content
app.layout = html.Div(
    [
        # Initialize the sidebar state to "open"
        dcc.Store(id="sidebar-state", data="close"),
        dcc.Location(id="url"),
        header,
        sidebar,
        content
    ]
)
app.layout.total_pages = 1
app.layout.jobs_per_page = 10

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
            html.H3("Upload Data", style={"textAlign": "center"}, className="custom-h3 mb-3"),

            dcc.Upload(
            id='upload-data',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select an Excel File', style={'color': 'var(--background-color)', 'textDecoration': 'underline'})
            ]),
            style={
                'width': '50%',
                'height': '50px',
                'lineHeight': '50px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': 'auto',
                'backgroundColor': 'var(--accent-color)',
                'color': 'var(--background-color)',
                'cursor': 'pointer'
            },
            multiple=False
        ),

            html.Div(id='output-data-upload'),
            html.Button(
                "Add Row",
                id="add-row-btn",
                n_clicks=0,
                style={  # Custom style
                    "display": "none",  # Make the button visible
                    "margin": "20px auto",
                    "backgroundColor": "transparent",  # Button background color
                    "color": "var(--knoppen-blauw)",  # Button text color
                    "border": "1px solid #7E909A",  # Border color
                    "padding": "5px 20px",  # Padding inside the button
                    "borderRadius": "0px",  # No rounded corners
                    "fontSize": "16px",  # Font size
                    "fontWeight": "bold",  # Bold text
                    "fontFamily": "montserrat, sans-serif",  # Font family
                    "cursor": "pointer",  # Pointer cursor on hover
                    "transition": "background-color 0.3s ease",  # Smooth hover transition
                }
            ),
            html.Div("", id='placeholder'),],
        style={  # Moved the style attribute here
            'backgroundColor': '#FFF',
            'display': 'none',
            'padding': '0px 20px',
            'color': 'black',
            'display': 'flex',
            'flexDirection': 'column',
            'justifyContent': 'space-between',
            'gap': '20px',
        },
    )


@app.callback(
    Output('output-data-upload', 'children'),
    Output('add-row-btn', 'style'),
    Input("file-input", "n_clicks"),
)
def restore_data(n_clicks):
    if n_clicks > 0 and app.layout.df is not None:
        df = app.layout.df
        return html.Div([dcc.Link(
                    dbc.Button(
                        "Submit Data",
                        id="submit-data-btn",
                        **button_style1
                    ),
                    href="/algorithm-settings",  # Link to the algorithm settings page
                    style={"textAlign": "center", "display": "block"}
                ),
            dash_table.DataTable(
                id="sortable-table",
                columns=[{"name": col, "id": col} for col in df.columns],
                data=df.to_dict("records"),
                sort_action="native",
                editable=True,
                row_deletable=True,
                **table_layout
            )
        ]), button_style2["style"]
    return html.Div("", style={'textAlign': 'center', 'font-family': 'Roboto'}), {"display": "none"}


# Callback to parse and display the uploaded file
@app.callback(
    [
        Output('output-data-upload', 'children', allow_duplicate=True),
        # Add output to modify button style
        Output('add-row-btn', 'style', allow_duplicate=True)
    ],
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    prevent_initial_call=True
)
def display_table(contents, filename):
    if contents is None:
        # If no file is uploaded, keep button hidden and display placeholder text
        return html.Div(
            "",
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
                process_time_titles = [
                    f"Process time {i+1}" for i in range(len(df.columns) - 4)]
            else:
                return html.Div(
                    "Unsupported file type. Please upload an Excel file with service times dedicated to each machine"
                ), {"display": "none"}

            titles = ['Job ID', 'Release Date',
                      'Due Date', 'Weight'] + process_time_titles
            df = df.set_axis(titles, axis=1)

            # Store the data in a global variable for graph updates
            app.layout.df = df  # Store the dataframe in the app layout

            # Return the data table and make the button visible
            return html.Div([
                dcc.Link(
                    dbc.Button(
                        "Submit Data",
                        id="submit-data-btn",
                        **button_style1
                    ),
                    href="/algorithm-settings",  # Link to the algorithm settings page
                    style={"textAlign": "center", "display": "block"}
                ),

                dash_table.DataTable(
                    id="sortable-table",
                    columns=[{"name": col, "id": col, "deletable": False}
                             for col in df.columns],
                    data=df.to_dict("records"),
                    sort_action="native",  # Enables sorting by clicking column headers
                    editable=True,
                    row_deletable=True,
                    **table_layout
                ),
            ]),  button_style2["style"]
        else:
            return html.Div("Unsupported file type. Please upload an Excel file."), {"display": "none"}
    except Exception as e:
        return html.Div(f"There was an error processing the file: {str(e)}"), {"display": "none"}


@app.callback(
    Output('sortable-table', 'data'),
    Input('sortable-table', 'data'),
    State('sortable-table', 'columns'),
    prevent_initial_call=True)
def update_input_data(rows, columns):
    for i, row in enumerate(rows):
        row[columns[0]['name']] = i + 1  # Assuming the first column is Job ID
    app.layout.df = pd.DataFrame(
        rows, columns=[col['name'] for col in columns]).astype(int)
    return rows # app.layout.df.iloc[-1]['Job ID']


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


def algorithm_settings_layout():
    return html.Div(
        [
            html.H3("Algorithm Settings", className="custom-h3 text-center mb-4"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label("Max Runtime:", className="mt-2 mb-2", id="parameter-label"),
                            dcc.Input(
                                id="max-runtime",
                                type="number",
                                placeholder="Enter value (default 100)",
                                className="form-control",
                            ),
                        ],
                        width=4,
                    ),
                    dbc.Col(
                        [ dbc.Row( html.Label("Choose Algorithm:", className="mt-2 mb-2", style={"display": "inline-block"})),
                          dbc.Row(  dbc.RadioItems(
                                id="algorithm-type",
                                className="radio-group",
                                inline=True,
                                inputClassName="btn-check",
                                labelClassName="btn btn-outline-primary",
                                labelCheckedClassName="active",
                                options=[
                                    {"label": "ILP", "value": 1},
                                    {"label": "Genetic", "value": 2},
                                ],
                                value=1,
                                style={"display": "inline-block", "margin": "0px 0px 0px 0px"},
                            )),
                        ],
                        width=4,
                    ),
                    dbc.Col(
                        [
                            html.Label("Algorithm Status:", className="mt-2 mb-2"),
                            html.Div("Not running yet", id='param-2'),
                        ],
                        width=4,
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Button(
                            "Run Algorithm", 
                            id='run-btn',
                            color="primary", 
                            className="mt-4", 
                            n_clicks=0,
                            style={  # Add custom style
                                "margin": "0px 0",
                                "backgroundColor": "var(--accent-color)",  # Button background color
                                "color": "#FFFFFF",  # Button text color
                                "border": "none",  # No border
                                "padding": "10px 20px",  # Padding inside the button
                                "borderRadius": "5px",  # Rounded corners
                                "fontSize": "16px",  # Font size
                                "fontWeight": "bold",  # Bold text
                                "fontFamily": "montserrat, sans-serif",  # Font family
                                "cursor": "pointer",  # Pointer cursor on hover
                                "transition": "background-color 0.3s ease",  # Smooth hover transition
                            }
                        ),
                        width="auto"
                    ),
                    dbc.Col(
                        html.Div(
                            [
                                dbc.Button("Download Solution", id="btn-algorithm-output",
                                           className="mt-3", style={"display": "none"}),
                                dcc.Download(id="schedule-excel"),
                            ]
                        ),
                        width="auto"
                    ),
                    dbc.Col(
                        dcc.Link(
                            dbc.Button(
                                "Show visualizations",
                                id="show-vis-btn",
                                className="mt-3",
                                style={"textAlign": "center",
                                       "display": "none"}
                            ),
                            href="/graphs",  # Link to the algorithm settings page
                        ),
                        width="auto"
                    ),
                ]
            ),
            dbc.Row([dbc.Col(html.Div("", id='results-table') ,width =12)])
        ]
    )


@app.callback(Output('parameter-label', 'children'),
              Output('max-runtime', 'placeholder'),
              Input('algorithm-type', 'value'),
              prevent_initial_call=True)
def update_parameters(algorithm_type):
    if algorithm_type == 1:
        return "Max Runtime:", "Enter value (default 100)"
    else:
        return "Max Generations:", "Enter value (default 100)"


@app.callback(Output('param-2', 'children'),
              Input('run-btn', 'n_clicks'),
              prevent_initial_call=True)
def is_algorithm_running(n_clicks):
    return f"Running"


@app.callback(
    Output('max-runtime', 'value'),
    Input('run-btn', 'n_clicks'),
    State('max-runtime', 'value'),
    prevent_initial_call=True
)
def enter_max_runtime_value(n_clicks, max_runtime):
    if max_runtime is None:
        max_runtime = 100
    return max_runtime


@app.callback(
    Output('param-2', 'children', allow_duplicate=True),
    Output('btn-algorithm-output', 'style'),
    Output('show-vis-btn', 'style'),
    Output('results-table', 'children'),
    Input('run-btn', 'n_clicks'),
    State('max-runtime', 'value'),
    State('algorithm-type', 'value'),
    prevent_initial_call=True
)
def run_specified_algorithm(n_clicks, max_runtime, algorithm_type):
    data = app.layout.df.copy()
    columns = ['job_id', 'release_date', 'due_date', 'weight'] + \
        [f"st_{i+1}" for i in range(len(data.columns)-4)]
    data.columns = columns

    if max_runtime is None:
        max_runtime = 100

    if algorithm_type == 2:
        schedule, score, runtime, _, _ = gen.runAlgorithmGen(data, 10, max_runtime)
    else:
        schedule, score, runtime = ilp.runAlgorithm(data, max_runtime)

    app.layout.schedule_df = schedule
    app.layout.schedule_stats = (score, runtime)

    schedule_table =  schedule = dash_table.DataTable(
            id="schedule-table",
            columns=[{"name": col, "id": col, "deletable": False}
                     for col in app.layout.schedule_df.columns],
            data=app.layout.schedule_df.to_dict("records"),
            # sort_action="native",  # Enables sorting by clicking column headers
            **table_layout
        )

    return f"Finished", button_style1["style"], button_style1["style"], schedule_table


@app.callback(
    Output('schedule-excel', 'data'),
    Input('btn-algorithm-output', 'n_clicks'),
    prevent_initial_call=True
)
def download_schedule(n_clicks):
    return dcc.send_data_frame(app.layout.schedule_df.to_excel, "schedule.xlsx", sheet_name="schedule")


def create_gannt_chart(schedule_df, highlight_job_id=None):
    fig = go.Figure()

    machines = int((len(schedule_df.columns)-1)/2)
    color_scheme = n_colors(
        'rgb(234, 106, 71)', 'rgb(0, 145, 213)', len(schedule_df), colortype='rgb')

    job_ids = schedule_df['Job ID'].tolist()
    job_colors = {job_id: color for job_id,
                  color in zip(job_ids, color_scheme)}

    for index, row in schedule_df.iterrows():
        for m in range(machines):
            opacity = 1 if (highlight_job_id is None or row['Job ID'] == highlight_job_id) else 0.3


            fig.add_trace(go.Bar(
                x=[row[f'Completion time machine {m+1}'] -
                    row[f'Start time machine {m+1}']],
                y=[m+1],
                base=row[f'Start time machine {m+1}'],
                orientation='h',
                marker_color=job_colors[row['Job ID']],
                opacity=opacity,  # Adjust opacity based on highlight_job_id
                showlegend=False,
                customdata=[row['Job ID']],  # Store the Job ID in customdata
                hovertemplate=(
                f"Machine: {m+1}<br>"
                f"Job ID: {int(row['Job ID'])}<br>"
                "<extra></extra>"
            )
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
        showlegend=False,
    )
    return fig


def create_secondary_gantt_chart(schedule_df, highlight_job_id=None):
    fig = go.Figure()

    machines = int((len(schedule_df.columns) - 1) / 2)
    machine_colors = n_colors(
        'rgb(255, 127, 80)', 'rgb(70, 130, 180)', machines, colortype='rgb')

    for index, row in schedule_df.iterrows():
        for m in range(machines):
            opacity = 1 if (highlight_job_id is None or row['Job ID'] == highlight_job_id) else 0.3

            fig.add_trace(go.Bar(
                x=[row[f'Completion time machine {m+1}'] - row[f'Start time machine {m+1}']],
                y=[row['Job ID']],
                base=row[f'Start time machine {m+1}'],
                orientation='h',
                marker_color=machine_colors[m],
                name=f"Machine {m+1}",
                opacity=opacity,  # Adjust opacity based on highlight_job_id
                #legendgroup=f"Machine {m+1}",
                customdata=[row['Job ID']],  # Store the Job ID in customdata
                hovertemplate=(
                    f"Job ID: {int(row['Job ID'])}<br>"
                    f"Machine: {m+1}<br>"
                    "<extra></extra>"
                ), 
                showlegend=False
            ))

    for job_id, color in enumerate(machine_colors):
        fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=10, color=color),
        name=f"Machine {job_id +1}",
    ))


    fig.update_layout(
        title="Machine Schedule for Jobs",
        xaxis_title="Time",
        yaxis_title="Job ID",
        barmode='stack',
        xaxis=dict(
            fixedrange=True  # Enable zooming for detailed view
        ),
        yaxis=dict(
            tickmode='linear',
            tick0=1,
            dtick=1,
            fixedrange=True  # Enable zooming for detailed view
        ),
        showlegend=False,
    )


    return fig



def create_runtime_and_score_display(runtime, score):
    runtime_card = dbc.Card(
        dbc.CardBody(
            [
                html.H4("Runtime", className="card-title text-center"),
                html.H2(f"{runtime:.2f}", className="card-text text-center"),
            ]
        ),
        style={"borderRadius": "10px", "margin": "10px", "backgroundColor": "white", "color": "#202020"},
    )

    score_card = dbc.Card(
        dbc.CardBody(
            [
                html.H4("Total weighted delay", className="card-title text-center"),
                html.H2(f"{score:.0f}", className="card-text text-center"),
            ]
        ),
        style={"borderRadius": "10px", "margin": "10px", "backgroundColor": "white", "color": "#202020"},
    )

    return runtime_card, score_card


# Graphs Section Layout
def graphs_layout():

    schedule_data = getattr(app.layout, 'schedule_df', None)
    if schedule_data is not None:

        score, runtime = app.layout.schedule_stats
        runtime_card, score_card = create_runtime_and_score_display(runtime, score)

        schedule_graph = dcc.Graph(
            id = 'gant_chart',
            figure=create_gannt_chart(schedule_data),     
            config={
                'staticPlot': False,  # Enable interactions
                'scrollZoom': True,   # Enable scrolling
                'displayModeBar': False,  # Show mode bar
            }
        )

        # Pagination controls
        total_jobs = len(schedule_data)
        jobs_per_page = app.layout.jobs_per_page     
        total_pages = (total_jobs + jobs_per_page - 1) // jobs_per_page
        app.layout.total_pages = total_pages

        if total_pages > 1:
            pagination_buttons = html.Div(
            [dbc.RadioItems(
            id="pagination-buttons",
            className="radio-group",
            inline=True,
            inputClassName="btn-check",
            labelClassName="btn btn-outline-primary",
            labelCheckedClassName="active",
            options=[{"label": f"{i+1}", "value": i+1} for i in range(total_pages)],
            value=1,
            style={"textAlign": "center", "margin": "20px auto"},
            )])
        else:
            pagination_buttons = html.Div("", style={"display": "none"})

        # Initial display of the first 10 jobs
        global initial_jobs
        initial_jobs = schedule_data.iloc[:jobs_per_page]

        schedule_graph2 = dcc.Graph(
            id="schedule-graph2",
            figure=create_secondary_gantt_chart(initial_jobs),
            config={
                'staticPlot': False,  # Enable interactions
                'scrollZoom': True,   # Enable scrolling
                'displayModeBar': False,  # Show mode bar
            }
        )

        page = html.Div("", id='page-number', style={"display": "none"})

        return html.Div(
            [
            html.H3("Stats and visualizations", className="custom-h3 text-center mb-3"),
            page,
            dbc.Row(
                [
                dbc.Col(runtime_card, width=6),
                dbc.Col(score_card, width=6)
                ]
            ),
            dbc.Row(
                [
                dbc.Col(schedule_graph, width=12),
                ]
            ),
                        dbc.Row(
                [
                dbc.Col(pagination_buttons, width=12, style={"textAlign": "center"}),
                ],
                style={"justifyContent": "center", "marginBottom": "20px"}
            ),
            dbc.Row(
                [
                dbc.Col(schedule_graph2, width=12),
                ]
            ),
            ]
        )

    return html.H3("The schedule will be displayed here")

@app.callback(
    Output('gant_chart', 'figure'),
    Input('gant_chart', 'clickData')  # Listen for clicks
)
def update_chart(click_data):
    schedule_df = getattr(app.layout, 'schedule_df', None)

    if click_data is None:
        return create_gannt_chart(schedule_df)  # No click, return the default chart

    # Get the Job ID from the clicked bar
    # clicked_job_id = click_data['points'][0]['text']   Assuming 'text' contains the Job ID
    # print(click_data)
    return create_gannt_chart(schedule_df, highlight_job_id=int(click_data['points'][0]['customdata'])) 


@app.callback(
    [Output("schedule-graph2", "figure"), Output("page-number", "style")],
    Input("pagination-buttons", "value"),
    prevent_initial_call=True
)
def update_graphs(selected_page):
    start_idx = (selected_page - 1) * app.layout.jobs_per_page
    end_idx = start_idx + app.layout.jobs_per_page
    page_jobs = app.layout.schedule_df.iloc[start_idx:end_idx]

    return create_secondary_gantt_chart(page_jobs), {"display" : None}


@app.callback(
    [Output("sidebar", "style"),
     Output("page-content", "style"),
     Output("hamburger-menu", "children"),  # Change icon of the hamburger menu
     # Modify the style of the hamburger menu
     Output("hamburger-menu", "style"),
     Output("sidebar-state", "data")],  # Store the sidebar's state (open/closed)
    Input("hamburger-menu", "n_clicks"),  # Triggered by the hamburger menu
    State("sidebar-state", "data"),
    prevent_initial_call=True  # Avoid triggering the callback initially
)
def toggle_sidebar(n_clicks, current_state):
    if current_state == "open":
        # Close the sidebar
        return (
            {"backgroundColor": "var(--header-color)", "padding-top": "20px", "height": "100vh",
                "width": "0px", "position": "fixed", "overflow": "hidden"},
            {"marginLeft": "0px", "padding-top": "100px",
                "backgroundColor": "var(--background-color)", "minHeight": "100vh", "color": "black"},
            "☰",  # Hamburger menu icon when sidebar is closed
            {"cursor": "pointer", "fontSize": "24px", "color": "white",
                "marginRight": "20px"},  # Hamburger menu style
            "closed"  # Update state to closed
        )
    else:
        # Open the sidebar
        return (
            {
                "backgroundColor": "var(--header-color)",
                "padding": "40px 0px",
                "height": "100vh",
                "width": "200px",  # Sidebar visible
                "position": "fixed",
                "overflow": "auto",
                "display": "flex",
                "flexDirection": "column",  # Stack items vertically
            },
            {"marginLeft": "200px", "padding": "100px 10px",
                "backgroundColor": "var(--background-color)", "minHeight": "100vh", "color": "black"},
            "×",  # Close icon when the sidebar is open
            {"cursor": "pointer", "fontSize": "26px", "lineheight": "1.2",
                "color": "white", "marginRight": "30px"},  # Hamburger menu style
            "open"  # Update state to open
        )


# Run the app
if __name__ == "__main__":
    app.run_server(debug=False)
