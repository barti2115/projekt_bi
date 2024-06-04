import pandas as pd
import sqlite3

# Function to pre-process data from SQLite database tables and store it in a CSV file
def pre_process_data():
    # Connect to the SQLite database
    conn = sqlite3.connect('data/offers.db')
    
    # Read data from the currencies table
    currencies = ["EUR", "USD"]
    
    # Iterate over the currency codes
    for currency_code in currencies:
        # Read data from the currency_code_exchange_rate_data table
        query = f"SELECT effective_date, rate FROM {currency_code}_exchange_rate_data"
        data = pd.read_sql_query(query, conn)
        
        # Convert the 'effective_date' column to datetime
        data['effective_date'] = pd.to_datetime(data['effective_date'])
        
        # Resample the data to daily frequency and interpolate missing values
        min_date = data['effective_date'].min()
        max_date = data['effective_date'].max()
        all_dates = pd.date_range(start=min_date, end=max_date, freq='D')
        data = data.set_index('effective_date').reindex(all_dates).interpolate(method='linear')
        data['rate'] = data['rate'].round(4)

        # Store pre-processed data in a table
        table_name = f'{currency_code}_pre_processed_data'
        
        # Check if the table exists
        cursor = conn.cursor()
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
        table_exists = cursor.fetchone()
        
        if table_exists:
            # Replace the existing table
            data.to_sql(table_name, conn, if_exists='replace')
        else:
            # Create a new table and store the data
            data.to_sql(table_name, conn)
    
    # Close the database connection
    conn.close()

if __name__ == "__main__":
    pre_process_data()
