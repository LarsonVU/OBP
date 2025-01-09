import dash
from dash import dcc, html, Input, Output, State, dash_table
import pandas as pd
import base64
import io

# Initialize the app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Excel Upload Dashboard"

# Layout of the app
app.layout = html.Div([
    html.H1("Excel File Uploader", style={"textAlign": "center"}),
    
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select an Excel File', style = {'color': 'blue'})
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
            df = df.set_axis(['Job ID', 'Release Date', 'Due Date', 'Weight', 'Service time M1', 'Service time M2', 'Service time M3'], axis=1)
            # Limit to first five rows
            #df = df.head(5)

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
                    editable= True,
                    row_deletable= True
                ),
            ],
        )
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

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
