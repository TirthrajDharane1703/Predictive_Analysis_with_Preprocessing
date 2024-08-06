import pandas as pd
import requests
from bs4 import BeautifulSoup
import base64
import time

def fetch_html(url, retries=3):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    }

    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            print(f"Error fetching HTML (Attempt {attempt + 1}/{retries}): {e}")
            time.sleep(2)  # Add a delay before retrying

    raise Exception("Failed to fetch HTML after multiple attempts.")

def convert_web_to_dataframe(web_url):
    try:
        html_content = fetch_html(web_url)
        soup = BeautifulSoup(html_content, 'html.parser')
        tables = soup.find_all('table')

        dfs_with_links = []
        for i, table in enumerate(tables):
            df = pd.read_html(str(table))[0]
            dfs_with_links.append((df, get_download_link(df, i + 1)))

            # Print the DataFrame (optional)
            print(f"Table {i + 1}:")
            print(df)

        return dfs_with_links if dfs_with_links else None
    except Exception as e:
        print(f"Error converting web page to DataFrame: {e}")
        return None

def get_download_link(df, table_num):
    csv_data = df.to_csv(index=False)
    b64 = base64.b64encode(csv_data.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="table_{table_num}.csv">Download Excel Table {table_num}</a>'
    return href
