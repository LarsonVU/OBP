from dash import Dash, dcc, html
import plotly.express as px
import dash_bootstrap_components as dbc

# Initialize the Dash app with the DARKLY theme
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

# Sample data for the graphs
df = px.data.iris()  # Iris dataset for example graphs

# Define a common layout for dark mode
dark_layout = {
    "plot_bgcolor": "#343a40",  # Match DARKLY theme background
    "paper_bgcolor": "#343a40",  # Match DARKLY theme background
    "font": {"color": "white"},  # White text for better contrast
}

# Create four example graphs with the dark layout
graph1 = dcc.Graph(
    figure=px.scatter(
        df, x="sepal_width", y="sepal_length", color="species", title="Scatter Plot"
    ).update_layout(dark_layout),
    config={"displayModeBar": False}  # Disable the mode bar
)

graph2 = dcc.Graph(
    figure=px.bar(
        df, x="species", y="sepal_width", title="Bar Chart"
    ).update_layout(dark_layout)
)

graph3 = dcc.Graph(
    figure=px.line(
        df, x="sepal_length", y="petal_length", title="Line Chart"
    ).update_layout(dark_layout)
)

graph4 = dcc.Graph(
    figure=px.histogram(
        df, x="petal_width", title="Histogram"
    ).update_layout(dark_layout)
)

# App layout with a 2x2 grid using Bootstrap
app.layout = html.Div(
    [
        dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(graph1, width=6),  # 6/12 width = half the row
                        dbc.Col(graph2, width=6),
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(graph3, width=6),
                        dbc.Col(graph4, width=6),
                    ]
                ),
            ],
            fluid=True,  # Make the layout responsive
        )
    ],
    style={
        "backgroundColor": "#343a40",  # Custom background color for the app
        "color": "white",  # White text for all text inside the Div
        "minHeight": "100vh",  # Ensure the Div covers the full viewport height
        "padding": "20px",  # Add some padding for better spacing
    },
)

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
