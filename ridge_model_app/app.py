import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sys
from model_preprocessing import load_model, preprocess_data, custom_tokenizer  # Make sure to import custom_tokenizer

# Ensure the tokenizer is available when loading vectorizers
sys.modules['custom_tokenizer'] = custom_tokenizer

# Set up the page layout
st.set_page_config(page_title="Ridge Model Application", layout="wide")

# Sidebar Menu
st.sidebar.title("Navigation")
option = st.sidebar.selectbox("Choose a section:", ["Upload Data", "Single Prediction", "About"])

st.title('Ridge Model Application')

if option == "Upload Data":
    st.subheader("Upload your dataset and get predictions from your Ridge model.")

    uploaded_file = st.file_uploader('Upload TSV or CSV File', type=['csv', 'tsv'])

    if uploaded_file is not None:
        # Detect file type and load accordingly
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file, delimiter='\t')

        st.write('Uploaded Data:')
        st.write(df.head())

        model = load_model()
        processed_data = preprocess_data(df)

        # Make predictions
        predictions = model.predict(processed_data)
        df['Predictions'] = np.expm1(predictions)  # Convert predictions back to original scale

        st.subheader("Predictions:")
        st.write(df.head())

        # Option to download predictions
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label='Download Predictions as CSV',
            data=csv,
            file_name='predictions.csv',
            mime='text/csv',
        )

elif option == "Single Prediction":
    st.subheader("Single Product Prediction")
    
    st.write("Enter the product details below and get a price prediction:")

    # Collecting user inputs
    name = st.text_input("Product Name")
    description = st.text_area("Product Description")
    brand_name = st.text_input("Brand Name")
    category_name = st.text_input("Category Name")
    shipping = st.selectbox("Shipping (0 = Buyer pays, 1 = Seller pays)", [0, 1])
    item_condition_id = st.selectbox("Item Condition (1 = New, 5 = Poor)", [1, 2, 3, 4, 5])

    if st.button("Predict Price"):
        user_data = pd.DataFrame({
            'name': [name],
            'item_description': [description],
            'brand_name': [brand_name],
            'category_name': [category_name],
            'shipping': [shipping],
            'item_condition_id': [item_condition_id]
        })

        model = load_model()
        processed_data = preprocess_data(user_data)
        prediction = model.predict(processed_data)
        predicted_price = np.expm1(prediction)[0]

        st.success(f"Predicted Price: ${predicted_price:.2f}")

elif option == "About":
    st.subheader("About This App")
    st.write("""
    This app predicts product prices using a trained Ridge Regression model.
    You can either upload a file with multiple products or manually enter a single product's details to get a prediction.
    """)




