from dash import Dash, dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import base64
import io
import plotly.express as px
import ilp_algorithm as ilp
import ilp_overtake as ilp_o
import plotly.graph_objects as go
import genetic as gen
import genetic_overtake as gen_o
from plotly.colors import n_colors
import dash
import os
import numpy as np

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
                            active="exact", style={"color": "#FFF"},  n_clicks=0, id="alg-settings"),
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

#Initialize global variables
app.layout.total_pages = 1
app.layout.jobs_per_page = 10
titles = ['Job ID', 'Release Date',
                      'Due Date', 'Weight'] + [f"Process time {i+1}" for i in range(3)]
app.layout.df = pd.DataFrame(0, index=range(5), columns=titles)
app.layout.df['Job ID'] = [i+1 for i in range(5)] 

# Callbacks to handle page navigation


@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname"),
)
def display_page(pathname):
    '''
    This function reads a URL path and returns the correct page layout.

    Input:
    - pathname: The URL path of the page

    Output:
    - Refer to the correct page layout
    '''
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
    '''
    This function creates the layout for the file input page.

    Output:
    - Page layout for the file input section
    '''

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
                            "Enter data manually", 
                            id='enter-data-btn',
                            n_clicks=0,
                            style={  # Add custom style
                                "margin": "20px auto",
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
                            }
                        ),
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


@app.callback(Output('output-data-upload', 'children',  allow_duplicate=True),
              Output('add-row-btn', 'style', allow_duplicate=True),
              Output('enter-data-btn', 'style', allow_duplicate=True),
              Input('enter-data-btn', 'n_clicks'),
              prevent_initial_call=True)
def enter_data(n_clicks):
    '''
    This function checks whether the user wants to enter data manually.

    Input:
    - clicks on the "Enter data manually" button

    Output:
    - Creates a table for manual data entry
    '''
    app.layout.df = pd.DataFrame(0, index=range(5), columns=titles)
    app.layout.df['Job ID'] = [i+1 for i in range(5)] 
    return html.Div([html.Div(
                    dbc.Button(
                        "Submit Data",
                        id="submit-data-btn",
                        href="/algorithm-settings",  # Link to the algorithm settings page
                        style={"textAlign": "center",} | button_style1["style"]
                    ), style={"display": "flex", "justifyContent": "center", "alignItems": "center"},),
            dash_table.DataTable(
                id="sortable-table",
                columns=[{"name": col, "id": col} for col in app.layout.df.columns],
                data=app.layout.df.to_dict("records"),
                sort_action="native",
                editable=True,
                row_deletable=True,
                **table_layout
            )
        ]), button_style2["style"], {"display": "none"}


@app.callback(
    Output('output-data-upload', 'children'),
    Output('add-row-btn', 'style'),
    Output('enter-data-btn', 'style'),
    Input("file-input", "n_clicks"),
)
def restore_data(n_clicks):
    '''
    This function restores the data on the input page, when returning from another page.
    Input:
    - n_clicks: number of times the "File Input" button is clicked

    Output:
    - Returns the data table and makes the buttons visible after returning from the file input page 
    '''
    if n_clicks > 0 and app.layout.df is not None:
        df = app.layout.df
        return html.Div([html.Div(
                    dbc.Button(
                        "Submit Data",
                        id="submit-data-btn",
                        href="/algorithm-settings",  # Link to the algorithm settings page
                        style={"textAlign": "center",} | button_style1["style"]
                    ), style={"display": "flex", "justifyContent": "center", "alignItems": "center"},),
            dash_table.DataTable(
                id="sortable-table",
                columns=[{"name": col, "id": col} for col in df.columns],
                data=df.to_dict("records"),
                sort_action="native",
                editable=True,
                row_deletable=True,
                **table_layout
            )
        ]), button_style2["style"],  {"display": "none"}
    return html.Div("", style={'textAlign': 'center', 'font-family': 'Roboto'}), {"display": "none"}, {  # Add custom style
                                "margin": "20px auto",
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
                            }


# Callback to parse and display the uploaded file
@app.callback(
    [
        Output('output-data-upload', 'children', allow_duplicate=True),
        # Add output to modify button style
        Output('add-row-btn', 'style', allow_duplicate=True),
        Output('enter-data-btn', 'style', allow_duplicate=True)
    ],
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    prevent_initial_call=True
)
def display_table(contents, filename):
    '''
    This function displays the table created from file input

    Input:
    - the contents of the uploaded file, and the filename

    Output:
    - Display the table with the uploaded data
    '''
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
            return html.Div([html.Div(
                    dbc.Button(
                        "Submit Data",
                        id="submit-data-btn",
                        href="/algorithm-settings",  # Link to the algorithm settings page
                        style={"textAlign": "center",} | button_style1["style"]
                    ), style={"display": "flex", "justifyContent": "center", "alignItems": "center"},),
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
            ]),  button_style2["style"], {"display": "none"}
        else:
            return html.Div("Unsupported file type. Please upload an Excel file."), {"display": "none"}, {"display": "block"}
    except Exception as e:
        return html.Div(f"There was an error processing the file: {str(e)}"), {"display": "none"}, {"display": "block"}


@app.callback(
    Output('sortable-table', 'data'),
    Input('sortable-table', 'data'),
    State('sortable-table', 'columns'),
    prevent_initial_call=True)
def update_input_data(rows, columns):
    '''
    This function updates the input data table to a valid format

    Input:
    - rows of the data table
    - column names of the data table

    Output:
    - Updated table to correct the data format
    '''
    default_value = 0  # Define the default value to replace faulty data
    
    for i, row in enumerate(rows):
        row[columns[0]['name']] = i + 1  # Assuming the first column is Job ID
        for col in columns:  # Validate each column value
            try:
                # Attempt to convert to int, if it's numeric data
                row[col['name']] = int(row[col['name']])
            except (ValueError, TypeError):
                # If conversion fails, replace with default value
                row[col['name']] = default_value
                
    app.layout.df = pd.DataFrame(
        rows, columns=[col['name'] for col in columns]).astype(int)
    return rows 


@app.callback(Output('sortable-table', 'data', allow_duplicate=True),
              Input('add-row-btn', 'n_clicks'),
              State('sortable-table', 'data'),
              State('sortable-table', 'columns'),
              prevent_initial_call=True
              )
def add_row(n_clicks, rows, columns):
    '''
    This function adds a new row to the input data table

    Input:
    - n_clicks: number of times the "Add Row" button is clicked
    - rows: data in the table
    - columns: column names of the table

    Output:
    - Updated table with a new row
    '''

    if rows is None:
        rows = []
    new_row = {col['id']: 0 for col in columns}  # Create an empty row
    rows.append(new_row)  # Append the new row

    return rows

@app.callback(Output('placeholder', 'children'),
              Input('submit-data-btn', 'n_clicks'),)
def reset_schedule(n_clicks):
    '''
    This function resets previous results when new data is entered

    Input:
    - n_clicks: number of times the "Submit Data" button is clicked

    Output:
    - Placeholder text (the previous schedule is removed)
    '''
    app.layout.schedule_df = None
    return html.Div("")


def algorithm_settings_layout():
    '''
    This function creates the layout for the algorithm settings page.

    Input:

    Output:
    - Page layout for the algorithm settings section
    '''
    algorithm_description = dbc.Col(
    dbc.Card(
        [
            dbc.CardHeader(
                html.Label("Algorithm Description:", className="fw-bold")
            ),
            dbc.CardBody(
                html.P(
                    "The integer linear program (ILP) will find the optimal solution given sufficient time. "
                    "However, finding a reasonable solution might take more time, especially on larger instances. Running this algorithm on instances with more than 500 jobs is not recommended. \
                      One can determine the maximum run time in seconds by filling in the parameter.",
                    className="mb-0", id="algorithm-description"
                ) ,             style={
        "height": "115px",  # Fixed height
        "overflow": "hidden",  # Hide overflowing text
        "textOverflow": "ellipsis",  # Add ellipsis if text overflows
        }
            )
,
        ],
        className="mt-2",
    ),
    width=8,
)
    parameter_input = dbc.Col([dbc.Row(
                        [
                            html.Label("Max Runtime:", className="mt-2 mb-2", id="parameter-label"),
                            dcc.Input(
                                id="max-runtime",
                                type="number",
                                placeholder="Enter value (default 100)",
                                className="form-control",
                            ),
                        ]), dcc.Input(id="population-size", type="number", placeholder="Enter value (default 10)", className="form-control", style={"display": "none"})],
                        width=6, id="parameters"
                    )

    allow_overtake_check = dcc.Checklist(
            id="overtake-checklist",
            options=[
                {"label": " Disable Overtake", "value": "disable"}
            ],
            inputClassName="checklist-item-unselected",
            labelClassName="checklist-item-unselected",
            style={"textAlign" : "center"}
        )

    return html.Div(
        [
            html.H3("Algorithm Settings", className="custom-h3 text-center mb-4"),
            dbc.Row([dbc.Col(
                        [ dbc.Row( html.Label("Choose Algorithm:", className="mt-2 mb-2", style={"display": "inline-block"})),
                          dbc.Row(  dbc.RadioItems(
                                id="algorithm-type",
                                className="radio-group",
                                #inline=True,
                                inputClassName="btn-check",
                                labelClassName="btn btn-outline-primary w-100",
                                labelCheckedClassName="active",
                                options=[
                                    {"label": "ILP", "value": 1},
                                    {"label": "Genetic", "value": 2},
                                ],
                                value=1,
                                style={"display": "inline-block", "margin": "0px 0px 0px 0px", },
                            )),
                            dbc.Row(allow_overtake_check),
                        ],
                        width=4,
                    align="stretch",), 
                    algorithm_description]),
                dbc.Row(
                [
                    parameter_input,
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
                            className="mt-3", 
                            n_clicks=0,
                            style={  # Add custom style
                                "margin": "20px 0",
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



@app.callback(Output("btn-algorithm-output", "style"),
             Output("show-vis-btn", "style"),
             Output("results-table", "children"), 
             Input("alg-settings", "n_clicks"),
            )
def restore_algorithm_settings(n_clicks):
    '''
    This function restores the algorithm settings when returning from another page.

    Input:
    - n_clicks: number of times the "Algorithm Settings" button is clicked

    Output:
    - Restores the algorithm settings and results if previously run
    '''
    sched_df = getattr(app.layout, 'schedule_df', None)
    if sched_df is not None:
        schedule_table =  dash_table.DataTable(
            id="schedule-table",
            columns=[{"name": col, "id": col, "deletable": False}
                     for col in app.layout.schedule_df.columns],
            data=app.layout.schedule_df.to_dict("records"),
            # sort_action="native",  # Enables sorting by clicking column headers
            **table_layout
        )
        return button_style1["style"], button_style1["style"], schedule_table
    return {"display": "none"}, {"display": "none"}, html.Div("")



@app.callback(Output('parameters', 'children'),
              Output('algorithm-description', 'children'),
              Input('algorithm-type', 'value'),
              prevent_initial_call=True)
def update_parameters(algorithm_type):
    '''
    This function updates the parameters and algorithm description based on the selected algorithm

    Input:
    - algorithm_type: the selected algorithm type

    Output:
    - Updated parameters
    - Algorithm description
    '''

    ILP_description = "The integer linear program (ILP) will find the optimal solution given sufficient time. \
                    However, finding a reasonable solution might take more time, especially on larger instances. Running this algorithm on instances with more than 500 jobs is not recommended. \
                      One can determine the maximum run time in seconds by filling in the parameter."
    Genetic_description = "The genetic algorithm will find a reasonable solution given a sufficient number of generations and population size. " \
                            "However, finding the optimal solution might take infinite generations. Additionally, the solution that the genetic \
                            algorithm gives is inherently random. One can enter the maximum number of generations and population size per generation below.\
                            Increasing these values will increase the runtime."

    ILP_parameters = dbc.Col(dbc.Row(
                        [
                            html.Label("Max Runtime:", className="mt-2 mb-2", id="parameter-label"),
                            dcc.Input(
                                id="max-runtime",
                                type="number",
                                placeholder="Enter value (default 100)",
                                className="form-control",
                            ),
                        dcc.Input(id="population-size", type="number", placeholder="Enter value (default 10)", className="form-control", style={"display": "none"})]),
                    )

    Genetic_parameters = dbc.Col(
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Label("Max Generations:", className="mt-2 mb-2", id="parameter-label"),
                        dcc.Input(
                            id="max-runtime",
                            type="number",
                            placeholder="Enter value (default 100)",
                            className="form-control",
                        ),
                    ],
                    width=6,  # Adjust width to fit half of the row
                ),
                dbc.Col(
                    [
                        html.Label("Population Size:", className="mt-2 mb-2", id="parameter-label2"),
                        dcc.Input(
                            id="population-size",
                            type="number",
                            placeholder="Enter value (default 10)",
                            className="form-control",
                        ),
                    ],
                    width=6,  # Adjust width to fit half of the row
                ),
            ]
        )
    )



    if algorithm_type == 1:
        return ILP_parameters, ILP_description
    else:
        return Genetic_parameters, Genetic_description


@app.callback(Output('param-2', 'children'),
              Input('run-btn', 'n_clicks'),
              prevent_initial_call=True)
def is_algorithm_running(n_clicks):
    '''
    This function shows that the algorithm is running

    Input:
    - n_clicks: number of times the "Run Algorithm" button is clicked

    Output:
    - A spinner and "Running" text
    '''
    return html.Div(
            [
                dbc.Spinner(size="sm", color="--knoppen-blauw"),
                " Running",
            ],
        )


@app.callback(
    Output('max-runtime', 'value'),
    Output('population-size', 'value'),
    Input('run-btn', 'n_clicks'),
    State('max-runtime', 'value'),
    State('population-size', 'value'),
    prevent_initial_call=True
)
def enter_max_runtime_value(n_clicks, max_runtime, pop_size):
    '''
    This function fills in the default values for the parameters if the user does not enter any values

    Input:
    - n_clicks: number of times the "Run Algorithm" button is clicked
    - max_runtime: the maximum runtime entered by the user
    - pop_size: the population size entered by the user

    Output:
    - Default value for the maximum runtime
    - Default value for the population size
    '''
    if max_runtime is None:
        max_runtime = 100
    if pop_size is None:
        pop_size = 10
    return max_runtime, pop_size


@app.callback(
    Output('param-2', 'children', allow_duplicate=True),
    Output('btn-algorithm-output', 'style', allow_duplicate=True),
    Output('show-vis-btn', 'style', allow_duplicate=True),
    Output('results-table', 'children',allow_duplicate=True),
    Input('run-btn', 'n_clicks'),
    State('max-runtime', 'value'),
    State('population-size', 'value'),
    State('algorithm-type', 'value'),
    State('overtake-checklist', 'value'),
    prevent_initial_call=True
)
def run_specified_algorithm(n_clicks, max_runtime, pop_size, algorithm_type, overtake):
    '''
    This function runs the algorithm with the specified parameters by the users

    Input:
    - n_clicks: number of times the "Run Algorithm" button is clicked
    - max_runtime: the maximum runtime entered by the user
    - pop_size: the population size entered by the user
    - algorithm_type: the selected algorithm type
    - overtake: whether overtaking is allowed

    Output:
    - Algorithm status as finished
    - Display button "download solution"
    - Display button "show visualizations"
    - Schedule table
    '''
    data = app.layout.df.copy()
    columns = ['job_id', 'release_date', 'due_date', 'weight'] + \
        [f"st_{i+1}" for i in range(len(data.columns)-4)]
    data.columns = columns

    if max_runtime is None:
        max_runtime = 100
    
    if pop_size is None:
        pop_size = 10

    if algorithm_type == 2:
        if not overtake:
            schedule, score, runtime, _, _ = gen_o.runAlgorithmGenO(data, pop_size, max_runtime)
        else:
            schedule, score, runtime, _, _ = gen.runAlgorithmGen(data, pop_size, max_runtime)
    else:
        if not overtake:
            schedule, score, runtime = ilp_o.runAlgorithm(data, max_runtime)
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


def create_secondary_gantt_chart(schedule_df, due_dates, highlight_job_id=None):
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

    # Add due date lines for each job
    for index, row in schedule_df.iterrows():
        fig.add_trace(go.Scatter(
            x=[due_dates.loc[index], due_dates.loc[index]],
            y=[row['Job ID'] - 0.4, row['Job ID'] + 0.4],
            mode="lines",
            line=dict(color="#8B0000", width=6),
            name=f"Due Dates",
            showlegend=(index % 10 == 0),  # Show legend only once
            hoverinfo="text",
            hovertext=f"Due Date for Job {int(row['Job ID'])}: {due_dates.loc[index]}",
        ))

    # Add legend for machines
    for job_id, color in enumerate(machine_colors):
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, color=color),
            name=f"Machine {job_id + 1}",
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
        showlegend=True,
        legend=dict(
        itemclick=False,  # Disable click interactions
        itemdoubleclick=False)  # Disable double-click interactions
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
            figure=create_secondary_gantt_chart(initial_jobs, app.layout.df["Due Date"]),
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

    return create_secondary_gantt_chart(page_jobs, app.layout.df["Due Date"]), {"display" : None}


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
local = True
if __name__ == "__main__":
    if local:
        app.run_server(debug=False)
    else:  
        port = int(os.environ.get("PORT", 10000))  # Render assigns a PORT dynamically
        app.run(host="0.0.0.0", port=port, debug=False)
