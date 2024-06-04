import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import sqlite3

# Create the Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    html.H1('Interactive Dashboard'),
    dcc.Dropdown(
        id='period-dropdown',
        options=[
            {'label': 'Week', 'value': 'week'},
            {'label': 'Month', 'value': 'month'},
            {'label': 'Year', 'value': 'year'},
            {'label': '10 Years', 'value': '10_years'},
            {'label': 'All Data', 'value': 'all_data'}
        ],
        value='week'
    ),
    dcc.Store(id='selected-currency', data='usd', storage_type='memory'),
    dcc.Dropdown(
        id='currency-dropdown',
        options=[
            {'label': 'USD', 'value': 'usd'},
            {'label': 'EUR', 'value': 'eur'}
        ],
        value='usd'
    ),
    dcc.Graph(id='data-plot'),
    dcc.Graph(id='predicted-plot'),
    html.Button('BUY', id='prediction-button')
])

# Define the callback functions

@app.callback(
    dash.dependencies.Output('data-plot', 'figure'),
    [dash.dependencies.Input('period-dropdown', 'value'),
     dash.dependencies.Input('currency-dropdown', 'value')]
)
def update_data_plot(selected_period, selected_currency):
    # Connect to the SQLite database
    conn = sqlite3.connect('data/offers.db')

    # Determine the table name based on the selected currency
    table_name = 'USD_pre_processed_data' if selected_currency == 'usd' else 'EUR_pre_processed_data'

    # Formulate SQL query based on the selected period
    if selected_period == 'week':
        query = f"SELECT * FROM {table_name} WHERE [index] >= date('now', '-7 days')"
    elif selected_period == 'month':
        query = f"SELECT * FROM {table_name} WHERE [index] >= date('now', '-1 month')"
    elif selected_period == 'year':
        query = f"SELECT * FROM {table_name} WHERE [index] >= date('now', '-1 year')"
    elif selected_period == '10_years':
        query = f"SELECT * FROM {table_name} WHERE [index] >= date('now', '-10 years')"
    else:
        query = f"SELECT * FROM {table_name}"

    # Read data from the database
    df = pd.read_sql_query(query, conn)

    # Close the database connection
    conn.close()

    # Create the data plot
    data_plot = {
        'data': [
            {'x': df['index'], 'y': df['rate'], 'type': 'line', 'name': 'Rate'}
        ],
        'layout': {
            'title': 'Data Plot'
        }
    }

    return data_plot

@app.callback(
    dash.dependencies.Output('predicted-plot', 'figure'),
    [dash.dependencies.Input('period-dropdown', 'value'),
     dash.dependencies.Input('currency-dropdown', 'value')]
)
def update_predicted_plot(selected_period, selected_currency):
    # Connect to the SQLite database
    conn = sqlite3.connect('data/offers.db')

    # Placeholder for table names, replace with actual table names
    table_name = 'future_values_usd' if selected_currency == 'usd' else 'future_values_eur'
    query = f"SELECT * FROM {table_name}"

    # Read data from the database
    df = pd.read_sql_query(query, conn)

    # Close the database connection
    conn.close()

    # Placeholder for predicted plot, replace this with actual prediction logic
    predicted_plot = {
        'data': [
            {'x': df['date'], 'y': df[f"{selected_currency}_prediction"], 'type': 'line', 'name': 'Predicted Rate'}
        ],
        'layout': {
            'title': 'Predicted Plot'
        }
    }

    # Close the database connection
    conn.close()

    return predicted_plot

@app.callback(
    dash.dependencies.Output('prediction-button', 'style'),
    [dash.dependencies.Input('predicted-plot', 'figure'),
     dash.dependencies.Input('currency-dropdown', 'value')]
)
def update_prediction_button(predicted_plot, selected_currency):
    # Connect to the SQLite database
    conn = sqlite3.connect('data/offers.db')

    # Determine the table name based on the selected currency
    table_name = 'USD_pre_processed_data' if selected_currency == 'usd' else 'EUR_pre_processed_data'
    query = f"SELECT * FROM {table_name}"

    # Read data from the database
    df = pd.read_sql_query(query, conn)

    # Close the database connection
    conn.close()
    # Placeholder for logic to check if predicted data is more than 5% than the latest value
    if any(y > 1.05 * df['rate'].iloc[-1] for y in predicted_plot['data'][0]['y']):
        return {'background-color': 'green'}
    else:
        return {'background-color': 'red'}

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
