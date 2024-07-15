import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import logging
import numpy as np
import dash_bootstrap_components as dbc

# Set up logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load and preprocess the data
df = pd.read_csv('data.csv')

# Aggregate data by console and publisher for the initial sunburst chart
agg_df = df.groupby(['console', 'publisher']).sum().reset_index()

# Create the initial scatter plot (with all data)
fig_scatter = px.scatter(
    df, x='critic_score', y='total_sales', color='genre',
    title='Critic Score vs. Total Sales',
    labels={'critic_score': 'Critic Score', 'total_sales': 'Total Sales'},
    hover_data=['title', 'developer', 'publisher', 'total_sales', 'genre', 'console']
)

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

def fill_null_values(df):
    null_count = df['critic_score'].isnull().sum() + df['total_sales'].isnull().sum()  # Count total null values before filling
    df.loc[df['critic_score'].isnull(), 'critic_score'] = np.round(np.random.uniform(1, 6, size=len(df[df['critic_score'].isnull()])), 2)
    df.loc[df['total_sales'].isnull(), 'total_sales'] = np.round(np.random.uniform(0.1, 2.5, size=len(df[df['total_sales'].isnull()])), 2)
    filled_null_count = null_count - df['critic_score'].isnull().sum() - df['total_sales'].isnull().sum()  # Count total null values filled
    logging.info(f"Filled {filled_null_count} null values.")
    return df

app.layout = html.Div(style={'backgroundColor': 'black'}, children=[
    html.H1("Video Game Sales Visualization Dashboard", style={'textAlign': 'center', 'font-family': ' Agency FB', 'font-style':'bold', 'color':'orange','font-size':'75px'}),
    html.Div(className='row', style={'backgroundColor': 'lightblack'}, children=[
        html.Div(className='col-md-6', children=[
            html.H2("Selection & filtering", style={'textAlign': 'center','font-family': ' Agency FB', 'color' :'orange'}),
            dcc.Dropdown(
                id='filter-dropdown',
                options=[
                    {'label': 'All', 'value': 'All'},
                    {'label': 'Console', 'value': 'console'},
                    {'label': 'Publisher', 'value': 'publisher'},
                    {'label': 'Developer', 'value': 'developer'},
                    {'label': 'Genre', 'value': 'genre'}
                ],
                value='All',
                style={'color': 'black'}
            ),
            html.Div(id='charts-container', children=[
                dcc.Graph(
                    id='console-publisher-sunburst',
                    figure=px.sunburst(agg_df, path=['console', 'publisher'], values='total_sales')
                ),
                html.Div(id='detail-chart-container')
            ])
        ]),
        html.Div(className='col-md-6', children=[
            html.H2("Visualization of genres", style={'textAlign': 'center', 'font-family': ' Agency FB','color':'red'}),
            dcc.Graph(id='scatter-plot', figure=fig_scatter),
            html.H2("Total sales details", style={'textAlign': 'center', 'font-family': ' Agency FB','color':'blue'}),
            html.Div(id='bar-chart-container'),
        ])
    ])
])

@app.callback(
    [Output('detail-chart-container', 'children'),
     Output('scatter-plot', 'figure'),
     Output('bar-chart-container', 'children'),
     Output('console-publisher-sunburst', 'figure')],
    [Input('console-publisher-sunburst', 'clickData'),
     Input('filter-dropdown', 'value')]
)
def update_charts(clickData, filter_value):
    logging.info(f"Filter selected: {filter_value}")

    # Start with the full dataset
    filtered_df = df.copy()

    # Apply additional filter if selected
    if filter_value and filter_value != 'All':
        filtered_df = filtered_df[filtered_df[filter_value].notnull()]
        logging.info(f"Filtered data based on {filter_value}. Number of records: {len(filtered_df)}")

    # Update sunburst chart with the filtered data based on the selected value
    if filter_value == 'console':
        filtered_sunburst_figure = px.sunburst(filtered_df, path=['console'], values='total_sales')
    elif filter_value == 'publisher':
        filtered_sunburst_figure = px.sunburst(filtered_df, path=['publisher'], values='total_sales')
    elif filter_value == 'developer':
        filtered_sunburst_figure = px.sunburst(filtered_df, path=['developer'], values='total_sales')
    elif filter_value == 'genre':
        filtered_sunburst_figure = px.sunburst(filtered_df, path=['genre'], values='total_sales')
    else:
        filtered_sunburst_figure = px.sunburst(filtered_df.groupby(['console', 'publisher']).sum().reset_index(), path=['console', 'publisher'], values='total_sales')

    # Handle sunburst chart clickData
    if clickData:
        path = clickData['points'][0]['id']
        logging.info(f"Clicked path: {path}")

        # Split the path into components
        path_parts = path.split('/')

        logging.info(f"Length of path_parts: {len(path_parts)}")

        # Create a filtering condition based on the path parts
        condition = pd.Series([True] * len(filtered_df))
        for i, part in enumerate(path_parts):
            if part:
                column = ['console', 'publisher', 'developer', 'title'][i]
                condition = condition & (filtered_df[column] == part)

        filtered_df = filtered_df[condition]

        logging.info(f"Filtered data after clickData. Number of records: {len(filtered_df)}")

        # Determine the path based on the depth of the click
        if len(path_parts) == 1:
            # Reset the detail chart and scatter plot when returning to the first level
            return None, fig_scatter, None, filtered_sunburst_figure
        elif len(path_parts) == 2:
            logging.info("Creating bar chart for total sales by title")
            sunburst_path = ['publisher', 'developer', 'title']
            bar_chart = px.bar(filtered_df, x='developer', y='total_sales', title='Total Sales by Title',labels={'total_sales': 'Total sales in Million', 'developer':'Developer'}, hover_data=['title','total_sales'])

            # Fill null values for the filtered DataFrame
            filtered_df = fill_null_values(filtered_df)

            detail_figure = px.sunburst(filtered_df, path=sunburst_path, values='total_sales')

            scatter_figure = px.scatter(
                filtered_df, x='critic_score', y='total_sales', color='genre',
                title='Critic Score vs. Total Sales',
                labels={'critic_score': 'Critic Score', 'total_sales': 'Total Sales'},
                hover_data=['title', 'developer', 'publisher', 'total_sales', 'genre']
            )

            return [dcc.Graph(id='detail-sunburst', figure=detail_figure),
                    scatter_figure,
                    dcc.Graph(id='detail-bar-chart', figure=bar_chart),
                    filtered_sunburst_figure]
    else:
        # Apply initial filter if no clickData is present
        filtered_scatter_figure = px.scatter(
            filtered_df, x='critic_score', y='total_sales', color='genre',
            title='Critic Score vs. Total Sales',
            labels={'critic_score': 'Critic Score', 'total_sales': 'Total Sales'},
            hover_data=['title', 'developer', 'publisher', 'total_sales', 'genre', 'console']
        )
        return None, filtered_scatter_figure, None, filtered_sunburst_figure

# Run the app
if __name__ == '__main__':
    app.run_server(debug=False,port=8051)
