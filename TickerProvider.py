import requests
from bs4 import BeautifulSoup

def get_most_active_tickers(num_tickers):
    """
    Fetches the most actively traded stocks by scraping Yahoo Finance.
    Returns a list of ticker symbols.
    """
    try:
        url = "https://finance.yahoo.com/most-active"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
        response = requests.get(url, headers=headers)
        response.raise_for_status() # Raise an exception for bad status codes

        soup = BeautifulSoup(response.text, 'html.parser')
        # Find the table containing the most active stocks
        table = soup.find('table')
        if not table:
            print("Could not find the data table on the Yahoo Finance page.")
            return []

        tickers = []
        # Find all rows in the table body
        for row in table.find('tbody').find_all('tr')[:num_tickers]:
            # The ticker symbol is in the first 'a' tag of the row
            ticker_element = row.find('a')
            if ticker_element:
                tickers.append(ticker_element.text)
        
        return tickers

    except Exception as e:
        print(f"An error occurred while fetching most active tickers: {e}")
        return []

if __name__ == "__main__":
    top_tickers = get_most_active_tickers()
    if top_tickers:
        print("Most Active Tickers:")
        for ticker in top_tickers:
            print(ticker)