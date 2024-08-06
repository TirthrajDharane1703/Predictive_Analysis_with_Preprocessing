from statistics import LinearRegression
import numpy as np
from sklearn.model_selection import train_test_split
import streamlit as st
from PIL import Image
import time
import streamlit as st
import pandas as pd
import base64
import time
from requests.exceptions import RequestException
from PyPDF2 import PdfReader
from sklearn.linear_model import LinearRegression
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error

# Importing functions from utils.py
from utils import convert_file_to_dataframe, get_num_pages

# Importing functions from converter files
from pdf_converter import convert_pdf_to_dataframe
from word_converter import convert_word_to_dataframe
from xml_converter import convert_xml_to_dataframe
from html_converter import convert_html_to_dataframe
from web_converter import convert_web_to_dataframe  # Update the import statement

def main():
    st.title('Prediction Analysis with Preprocessing')

    # Sidebar navigation
    st.sidebar.title('Navigation')
    page_selection = st.sidebar.radio("Go to", ('Data Conversion', 'Data Visualization', 'Prediction'))

    if page_selection == 'Data Conversion':
        data_conversion_page()
    elif page_selection == 'Data Visualization':
        data_visualization_page()
    elif page_selection == 'Prediction':
        prediction_page()

    # Add some elements below the navigation
    st.sidebar.title('About')
    st.sidebar.info(
        "This app allows you to perform prediction analysis "
        "with preprocessing. You can convert various data formats, "
        "visualize data, and make predictions."
    )

    st.sidebar.title('Contact')
    st.sidebar.info(
        "For any inquiries or support, please reach out to: \n"
        "tirthrajdharane1703@gmail.com\n"
        "neelkapadia22@gmail.com\n"
        "Vinitruparelia123@gmail.com\n"
        "Surajnate29@gmail.com\n"
    )

    st.sidebar.title('Version')
    st.sidebar.info("Version 1.0")
    
    

    
    
def data_conversion_page():
    st.header('Data Conversion')
    # Upload file
    uploaded_file = st.file_uploader("Upload a file", type=["pdf", "docx", "xml", "html"])
    if uploaded_file:
        progress_bar = st.progress(0)
        progress_status = st.empty()

        # Display progress bar while uploading
        for percent_complete in range(100):
            progress_bar.progress(percent_complete + 1)
            progress_status.text(f"Uploading... {percent_complete + 1}%")
            time.sleep(0.1)

        # Convert uploaded file to DataFrame
        file_name = uploaded_file.name.lower()  # Convert file name to lowercase
        file_type = file_name.split(".")[-1]  # Extract file extension
        if file_type == "pdf":
            num_pages = get_num_pages(uploaded_file)  # Get the number of pages for PDF files
            st.info(f"Number of pages in PDF file: {num_pages}")
        dfs = convert_file_to_dataframe(uploaded_file, file_type)

        # Display the DataFrames if not empty
        if dfs is not None:
            selected_tables = st.multiselect("Select tables to download:", range(1, len(dfs) + 1),
                                             default=range(1, len(dfs) + 1))

            for i, df in enumerate(dfs):
                # Check if the user selected this table for download
                if i + 1 in selected_tables:
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        # Display the DataFrame content
                        st.write(f"Converted DataFrame from table {i + 1}:")
                        st.write(df)

                        # Download button for the converted Excel file
                        if st.button(f"Download Excel Table {i + 1}"):
                            csv_data = df.to_csv(index=False)
                            b64 = base64.b64encode(csv_data.encode()).decode()
                            href = f'<a href="data:file/csv;base64,{b64}" download="converted_table_{i + 1}.csv">Download Excel Table {i + 1}</a>'
                            st.markdown(href, unsafe_allow_html=True)
                    elif isinstance(df, str):
                        st.warning(f"Table {i + 1} is empty.")

    st.title("Web Page to DataFrame")
    web_url = st.text_input("Enter the URL of the web page:")
    if st.button("Convert"):
        if web_url:
            st.write("Converting web page to DataFrame...")
            dfs_with_links = convert_web_to_dataframe(web_url)
            if dfs_with_links:
                for i, (df, download_link) in enumerate(dfs_with_links):
                    st.write(f"Table {i + 1}:")
                    st.write(df)
                    st.markdown(download_link, unsafe_allow_html=True)


def data_visualization_page():
    st.header('Data Visualization')
    st.title('Medical Transaction Data Visualization')

    uploaded_file = st.file_uploader("Upload medical data CSV file", type=['csv'])
    target_column = st.text_input("Enter the target column name")

    if uploaded_file is not None and target_column:
        # Read the CSV file and preprocess the data
        data = preprocess_data(uploaded_file)
        st.write('Preprocessed Data:')
        st.write(data.head())
        
        if data is not None:
            st.write('Data preview:')
            st.write(data.head())

            st.write('Performing feature selection...')
            selected_data = feature_selection(data, target_column)

            if selected_data is not None:
                st.write('Selected features:')
                st.write(selected_data.head())

                st.write('Visualizing data...')
                plot_filename = visualize_data(selected_data)

                # Save and download visualization as PDF
                st.write('Plots saved as PDF:')
                download_pdf(plot_filename)
def preprocess_data(file):
    # Read the CSV file, skipping the first four rows
    data = pd.read_csv(file, skiprows=4)
    
    # Print out the first few rows of the dataset to verify if the headers are loaded correctly
    print("First few rows of the dataset:")
    print(data.head())
    
    # Check if 'InvDate' column exists in the dataframe
    if 'InvDate' in data.columns:
        # Convert 'InvDate' column to datetime format
        data['InvDate'] = pd.to_datetime(data['InvDate'], format='%m/%d/%Y')
        # Extract month feature
        data['month'] = data['InvDate'].dt.month
        # Extract year feature
        data['year'] = data['InvDate'].dt.year
        # Drop original 'InvDate' column
        data.drop(columns=['InvDate'], inplace=True)
    else:
        st.error("Column 'InvDate' not found in the dataset. Please check the column name and try again.")
        return None
    
    # Remove 'GInvno' column if it exists
    if 'GInvNo' in data.columns:
        data.drop(columns=['GInvNo'], inplace=True)
    
    return data
def feature_selection(data, target_column):
    if target_column not in data.columns:
        st.error(f"Target column '{target_column}' not found in the dataset. Please provide a valid target column.")
        return None
    if 'Ledger Account' in data.columns:
        data.drop(columns=['Ledger Account'], inplace=True)

    # Skip the last row
    data = data.iloc[:-1]

    # Separate categorical and numerical columns
    categorical_cols = data.select_dtypes(include=['object']).columns
    numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns

    # One-hot encode categorical columns
    if len(categorical_cols) > 0:
        encoder = OneHotEncoder(drop='first')
        encoded_cols = pd.DataFrame(encoder.fit_transform(data[categorical_cols]).toarray())
        encoded_cols.columns = encoder.get_feature_names_out(categorical_cols)
        data = pd.concat([data.drop(columns=categorical_cols), encoded_cols], axis=1)

    # Split features and target variable
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Initialize Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=100, random_state=42)

    # Fit the model
    rf.fit(X, y)

    # Feature importance
    feature_importances = pd.Series(rf.feature_importances_, index=X.columns)
    important_features = feature_importances.nlargest(7).index.tolist()  # Select top 7 important features
 
    # Ensure 'Area' column is included in the selected data
    if 'Area' in data.columns:
        if 'Area' not in important_features:
            important_features.append('Area')

    # Select relevant columns
    selected_data = data[important_features + [target_column]]

    return selected_data
def visualize_data(data):
    print("Column names in selected_data:", data.columns)  # Debugging statement
    
    # Initialize a PDF file
    plot_filename = 'medical_data_plots.pdf'
    with PdfPages(plot_filename) as pdf:
        # Scatter Plot
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=data, x=data.columns[0], y=data.columns[1])
        plt.title(f'Scatter Plot: {data.columns[0]} vs {data.columns[1]}')
        plt.xlabel(data.columns[0])
        plt.ylabel(data.columns[1])
        pdf.savefig()
        plt.close()

        # Histogram Plot
        plt.figure(figsize=(8, 6))
        sns.histplot(data=data, x=data.columns[2], bins=20, kde=True)
        plt.title(f'Histogram Plot: {data.columns[2]} Distribution')
        plt.xlabel(data.columns[2])
        pdf.savefig()
        plt.close()

        # Line Graph
        plt.figure(figsize=(8, 6))
        sns.lineplot(data=data, x=data.columns[0], y=data.columns[1])
        plt.title(f'Line Graph: {data.columns[0]} vs {data.columns[1]}')
        plt.xlabel(data.columns[0])
        plt.ylabel(data.columns[1])
        pdf.savefig()
        plt.close()
#Currently implementing
        # Radar Graph for Sales Distribution Area-wise
        if 'Area' in data.columns:  # Check if 'Area' column exists
            # Group by 'Area' and calculate total sales quantity
            sales_by_area = data.groupby('Area')['Qty'].sum().reset_index()
            # Plot radar graph
            plt.figure(figsize=(10, 8))
            for index, row in sales_by_area.iterrows():
                plt.plot(row.drop('Area'), label=row['Area'])
            plt.title('Sales Distribution Area-wise (Radar Graph)')
            plt.xlabel('Month')
            plt.ylabel('Total Sales Quantity')
            plt.legend()
            pdf.savefig()
            plt.close()
        else:
            print("Column 'Area' not found in selected_data.")  # Debugging statement
#End of Currently implementing Code block
        # Heatmap
        plt.figure(figsize=(10, 8))
        correlation_matrix = data.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        plt.title('Heatmap: Correlation Matrix')
        pdf.savefig()
        plt.close()

    return plot_filename
def download_pdf(filename):
    with open(filename, 'rb') as f:
        pdf_bytes = f.read()
    st.download_button(label='Download PDF', data=pdf_bytes, file_name=filename, mime='application/pdf')


def prediction_page():

    # Upload medical data CSV file
    uploaded_file = st.file_uploader("Upload medical data CSV file", type=['csv'])
    target_column = st.text_input("Enter the target column name")

    if uploaded_file is not None and target_column:
        # Preprocess data and perform feature selection
        data = preprocess_data(uploaded_file)
        selected_data = feature_selection(data, target_column)

        if selected_data is not None:
            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(selected_data.drop(columns=[target_column]), selected_data[target_column], test_size=0.2, random_state=42)
            
            # Train the model
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)

            # Make prediction using test data
            y_pred = rf.predict(X_test)

            # Calculate Mean Squared Error
            mse = mean_squared_error(y_test, y_pred)

            # Display MSE in red color
            st.subheader("Mean Squared Error (MSE)")
            st.write(f"<span style='color:red'>{mse:.2f}</span>", unsafe_allow_html=True)

            # Combine predicted and actual values into a DataFrame
            result_df = pd.DataFrame({'Predicted': y_pred, 'Actual': y_test}).reset_index(drop=True)

            # Display predicted and actual values in a single table
            st.subheader("Predicted vs Actual Values")
            st.table(result_df)

            # Plot predicted values as a bar chart
            st.subheader("Bar Plot of Predicted Values")
            plt.figure(figsize=(10, 6))
            plt.bar(range(len(y_pred)), y_pred, color='skyblue', label='Predicted')
            plt.xlabel('Index')
            plt.ylabel('Predicted Value')
            plt.title('Bar Plot of Predicted Values')
            st.pyplot(plt)  # Display the plot in Streamlit

if __name__ == "__main__":
    main()
