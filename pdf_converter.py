import pandas as pd
from tabula.io import read_pdf

def convert_pdf_to_dataframe(pdf_file):
    try:
        tables = read_pdf(pdf_file, pages="all", multiple_tables=True)
        dfs = []
        for table in tables:
            df = pd.DataFrame(table)
            dfs.append(df)
        return dfs if dfs else None
    except Exception as e:
        return None
