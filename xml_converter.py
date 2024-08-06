import pandas as pd
from bs4 import BeautifulSoup

def convert_xml_to_dataframe(xml_content):
    try:
        soup = BeautifulSoup(xml_content, "xml")
        tables = soup.find_all("table")
        dfs = []
        for table in tables:
            data = []
            headers = []
            for row in table.find_all("row"):
                row_data = [cell.get_text(strip=True) for cell in row.find_all("cell")]
                data.append(row_data)
                if not headers:
                    headers = [f"Header_{i+1}" for i in range(len(row_data))]
            df = pd.DataFrame(data, columns=headers)
            dfs.append(df)
        return dfs if dfs else None
    except Exception as e:
        return None
