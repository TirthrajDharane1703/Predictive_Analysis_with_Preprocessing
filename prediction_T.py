import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load the medical sales data
def load_data(file):
    data = pd.read_csv(file)
    return data

# Perform feature selection (if needed)
def select_features(data):
    # Modify feature selection based on your dataset
    selected_features = data[['feature1', 'feature2', 'feature3']]  
    return selected_features

# Train the Random Forest model
def train_model(X_train, y_train):
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    return rf

# Make predictions
def make_predictions(model, X_test):
    predictions = model.predict(X_test)
    return predictions

def main():
    st.title("Medical Sales Prediction")

    # File upload
    uploaded_file = st.file_uploader("Upload medical sales data (CSV file)", type=['csv'])

    if uploaded_file is not None:
        st.write("File uploaded successfully!")

        # Load data
        data = load_data(uploaded_file)

        # Display data
        st.subheader("Data Preview:")
        st.write(data.head())

        # Check if data is loaded successfully
        if not data.empty:
            # Select features
            selected_features = select_features(data)

            # Display selected features
            st.subheader("Selected Features:")
            st.write(selected_features)

            # Get the target column label from the user
            target_column = st.text_input("Enter the target column label")

            if target_column:
                if target_column in data.columns:  # Check if target column is present in the dataset
                    # Split data into features and target variable
                    X = selected_features
                    y = data[target_column]

                    # Split the data into training and testing sets
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                    # Train the model
                    model = train_model(X_train, y_train)

                    # Make predictions
                    predictions = make_predictions(model, X_test)

                    # Display predictions
                    st.subheader("Predictions:")
                    st.write(predictions)
                else:
                    st.error(f"Target column '{target_column}' not found in the dataset. Please enter a valid target column label.")
        else:
            st.error("Failed to load data. Please upload a valid CSV file.")

if __name__ == "__main__":
    main()
