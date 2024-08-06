import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from matplotlib.backends.backend_pdf import PdfPages
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def preprocess_data(file):
    data = pd.read_csv(file)
    # Convert date column to datetime format
    data['date'] = pd.to_datetime(data['date'])
    # Extract month feature
    data['month'] = data['date'].dt.month
    # Drop original date column
    data.drop(columns=['date'], inplace=True)
    return data

def feature_selection(data, target_column):
    if target_column not in data.columns:
        st.error(f"Target column '{target_column}' not found in the dataset. Please provide a valid target column.")
        return None

    # Drop 'datanum' column
    data.drop(columns=['datanum'], inplace=True, errors='ignore')

    # Split features and target variable
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Initialize Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=100, random_state=42)

    # Fit the model
    rf.fit(X, y)

    # Feature importance
    feature_importances = pd.Series(rf.feature_importances_, index=X.columns)
    important_features = feature_importances.nlargest(3).index.tolist()  # Select top 3 important features

    # Select relevant columns
    selected_data = data[important_features + [target_column]]

    return selected_data

def visualize_data(data):
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

        # Heatmap
        plt.figure(figsize=(10, 8))
        correlation_matrix = data.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        plt.title('Heatmap: Correlation Matrix')
        pdf.savefig()
        plt.close()

    return plot_filename

def train_xgboost(data, target_column):
    # Split features and target variable
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize XGBoost regressor
    xgb = XGBRegressor()

    # Train the model
    xgb.fit(X_train, y_train)

    # Make predictions
    y_pred_train = xgb.predict(X_train)
    y_pred_test = xgb.predict(X_test)

    # Evaluate model performance
    train_rmse = mean_squared_error(y_train, y_pred_train, squared=False)
    test_rmse = mean_squared_error(y_test, y_pred_test, squared=False)

    return xgb, train_rmse, test_rmse

def predict_future_expenses(model, data):
  def predict_future_expenses(model, data):
    # Assuming the last row in the data is the most recent observation
    recent_observation = data.iloc[-1].copy()

    # Increment month
    recent_observation['date'] += pd.DateOffset(months=1)

    # Predict future expenses
    future_expenses = model.predict(recent_observation.values.reshape(1, -1))

    return future_expenses

def download_pdf(filename):
    with open(filename, 'rb') as f:
        pdf_bytes = f.read()
    st.download_button(label='Download PDF', data=pdf_bytes, file_name=filename, mime='application/pdf')

def main():
    st.title('Medical Transaction Data Visualization and Future Expenses Prediction')

    uploaded_file = st.file_uploader("Upload medical data CSV file", type=['csv'])
    target_column = st.text_input("Enter the target column name")

    if uploaded_file is not None and target_column:
        data = preprocess_data(uploaded_file)
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

            st.write('Training XGBoost model...')
            model, train_rmse, test_rmse = train_xgboost(selected_data, target_column)
            st.write(f'Training RMSE: {train_rmse}')
            st.write(f'Test RMSE: {test_rmse}')

            st.write('Model training complete!')

            st.write('Predicting future expenses...')
            future_expenses = predict_future_expenses(model, selected_data)
            st.write('Predicted future expenses:', future_expenses)

if __name__ == "__main__":
    main()
