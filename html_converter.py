import pandas as pd
from bs4 import BeautifulSoup

def convert_html_to_dataframe(html_content):
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        tables = soup.find_all('table')
        dfs = []
        for table in tables:
            df = pd.read_html(str(table))[0]
            dfs.append(df)
        return dfs if dfs else None
    except Exception as e:
        return None
