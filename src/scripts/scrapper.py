import requests
import sqlite3
import datetime


def get_response_contents(url):
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.json()

def get_all_data():
    conn = sqlite3.connect('data/offers.db')
    c = conn.cursor()

    currencies = ["EUR", "USD"]
    for currency in currencies:
        c.execute(f'''
            CREATE TABLE IF NOT EXISTS {currency}_exchange_rate_data (
                effective_date TEXT PRIMARY KEY,
                rate REAL
            )
        ''')

    response=[]
    startDate = "2002-01-02"
    endDate = datetime.date.today().strftime("%Y-%m-%d")

    # Divide the time period into parts of 367 days
    start_date = datetime.datetime.strptime(startDate, "%Y-%m-%d")
    end_date = datetime.datetime.strptime(endDate, "%Y-%m-%d")
    delta = datetime.timedelta(days=367)
    date_range = []
    while start_date <= end_date:
        if start_date + delta > end_date:
            date_range.append((start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")))
        else:
            date_range.append((start_date.strftime("%Y-%m-%d"), (start_date + delta).strftime("%Y-%m-%d")))
        start_date += delta

    response = []
    for currency in currencies:
        for start, end in date_range:
            response.append(get_response_contents(f"http://api.nbp.pl/api/exchangerates/rates/A/{currency}/{start}/{end}/"))

    # Store all the responses in a table
    conn = sqlite3.connect('data/offers.db')
    c = conn.cursor()
    for currency_entry in response:
        for rate_entry in currency_entry['rates']:
                c.execute(f'''
                    INSERT OR IGNORE INTO {currency_entry['code']}_exchange_rate_data VALUES (?, ?)
                ''', (rate_entry['effectiveDate'], rate_entry['mid']))
    conn.commit()
    conn.close()

def main():
   get_all_data()

if __name__ == "__main__":
    main()