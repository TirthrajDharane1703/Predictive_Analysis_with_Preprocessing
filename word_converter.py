import pandas as pd
from docx import Document
import warnings

def convert_word_to_dataframe(docx_file):
    try:
        doc = Document(docx_file)
        tables = []
        for table in doc.tables:
            data = []
            for row in table.rows:
                row_data = []
                for cell in row.cells:
                    row_data.append(cell.text)
                data.append(row_data)
            df = pd.DataFrame(data)
            tables.append(df)
        return tables if tables else None
    except Exception as e:
        return None
